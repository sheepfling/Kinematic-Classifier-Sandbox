from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, log, pi, sqrt
import argparse
import csv
import io
import os
import random
from pathlib import Path


FEATURE_NAMES = (
    "reverse_motion",
    "high_speed",
    "positive_accel",
    "hard_brake",
    "oscillatory",
    "near_envelope",
)

PHASE_NAMES = (
    "steady",
    "braking",
    "powered",
    "reversing",
    "oscillating",
    "edge",
)

FEATURE_DESCRIPTIONS = {
    "reverse_motion": "sustained negative velocity or backtracking motion",
    "high_speed": "speed magnitude beyond the nominal low-speed envelope",
    "positive_accel": "persistent positive acceleration or thrust-like drive",
    "hard_brake": "persistent negative acceleration or stopping pressure",
    "oscillatory": "sign changes or alternating acceleration regimes",
    "near_envelope": "motion near the class speed or acceleration envelope edge",
}

CLASS_FEATURE_PRIORS = {
    "coast": ("steady",),
    "drift": ("reverse_motion",),
    "brake": ("hard_brake",),
    "maneuver": ("oscillatory", "reverse_motion"),
    "powered": ("high_speed", "positive_accel"),
    "unknown": ("high_speed", "oscillatory", "near_envelope"),
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def gaussian_interval_probability(mean: float, variance: float, limit: float) -> float:
    if limit <= 0.0:
        return 0.0
    safe_variance = max(variance, 1e-9)
    sigma = sqrt(safe_variance)
    upper = (limit - mean) / sigma
    lower = (-limit - mean) / sigma
    probability = _normal_cdf(upper) - _normal_cdf(lower)
    return _clamp(probability, 0.0, 1.0)


def _gaussian_upper_tail_probability(mean: float, variance: float, threshold: float) -> float:
    safe_variance = max(variance, 1e-9)
    sigma = sqrt(safe_variance)
    return _clamp(1.0 - _normal_cdf((threshold - mean) / sigma), 0.0, 1.0)


def _gaussian_lower_tail_probability(mean: float, variance: float, threshold: float) -> float:
    safe_variance = max(variance, 1e-9)
    sigma = sqrt(safe_variance)
    return _clamp(_normal_cdf((threshold - mean) / sigma), 0.0, 1.0)


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values))


def _gaussian_logpdf(value: float, mean: float, variance: float) -> float:
    safe_variance = max(variance, 1e-9)
    return -0.5 * (log(2.0 * pi * safe_variance) + ((value - mean) ** 2) / safe_variance)


def _entropy(weights: dict[str, float], epsilon: float = 1e-12) -> float:
    return -sum(value * log(max(value, epsilon)) for value in weights.values())


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = _clamp(q, 0.0, 1.0) * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _transition_matrix(dt: float) -> tuple[tuple[float, float, float], ...]:
    dt_sq = dt * dt
    return (
        (1.0, dt, 0.5 * dt_sq),
        (0.0, 1.0, dt),
        (0.0, 0.0, 1.0),
    )


def _matvec3(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(row[index] * vector[index] for index in range(3)) for row in matrix)  # type: ignore[return-value]


def _matmul3(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][inner] * right[inner][col] for inner in range(3)) for col in range(3))
        for row in range(3)
    )


def _transpose3(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))


def _add_vec3(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _add_mat3(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(left[row][col] + right[row][col] for col in range(3))
        for row in range(3)
    )


@dataclass(frozen=True, slots=True)
class ToyClassSpec:
    name: str
    accel_bias: float
    jerk_sigma: float
    accel_limit: float
    speed_limit: float
    prior_weight: float


@dataclass(frozen=True, slots=True)
class ToyScenarioSpec:
    name: str
    class_name: str
    velocity_range: tuple[float, float]
    accel_range: tuple[float, float]
    phase_targets: tuple[tuple[float, float, str], ...] = ()
    jerk_scale: float = 1.0
    accel_bias_scale: float = 1.0
    maneuver_impulse: float = 0.55
    flip_period_divisor: int = 6
    velocity_clip_scale: float = 0.9
    accel_clip_scale: float = 0.9
    unknown_bias: float = 1.4
    obs_sigma_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class ToyTrack:
    class_name: str
    scenario_name: str
    true_features: tuple[str, ...]
    true_phase_labels: tuple[str, ...]
    target_phase_labels: tuple[str, ...]
    seed: int
    dt: float
    positions_true: tuple[float, ...]
    velocities_true: tuple[float, ...]
    accelerations_true: tuple[float, ...]
    positions_obs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ToyDataset:
    tracks: tuple[ToyTrack, ...]
    class_specs: tuple[ToyClassSpec, ...]
    scenario_specs: tuple[ToyScenarioSpec, ...]
    steps: int
    dt: float
    obs_sigma: float
    seed: int


@dataclass(frozen=True, slots=True)
class ToyBenchmarkConfig:
    steps: int = 32
    dt: float = 1.0
    obs_sigma: float = 0.75
    lambda_v: float = 0.6
    lambda_a: float = 0.4
    lambda_velocity_center: float = 0.14
    lambda_accel_center: float = 0.10
    lambda_direction_match: float = 0.18
    lambda_oscillation_match: float = 0.16
    lambda_observed_velocity: float = 0.12
    lambda_observed_accel: float = 0.10
    epsilon: float = 1e-9
    unknown_log_penalty: float = 0.55
    initial_position_variance: float = 1.0
    initial_speed_variance: float = 16.0
    initial_accel_variance: float = 4.0
    representative_trace_steps: int = 5
    feature_detection_threshold: float = 0.55
    class_specs: tuple[ToyClassSpec, ...] = field(default_factory=lambda: default_class_specs())
    scenario_specs: tuple[ToyScenarioSpec, ...] = field(default_factory=lambda: default_toy_scenarios())


@dataclass(frozen=True, slots=True)
class ClassPosteriorStep:
    predicted_class_weights: dict[str, float]
    updated_class_weights: dict[str, float]
    log_likelihood_terms: dict[str, dict[str, float]]
    feature_probabilities: dict[str, float]
    detected_phase_label: str
    posterior_entropy: float
    map_class: str


@dataclass(frozen=True, slots=True)
class ClassificationRun:
    true_class: str
    scenario_name: str
    true_features: tuple[str, ...]
    true_phase_labels: tuple[str, ...]
    target_phase_labels: tuple[str, ...]
    detected_features: tuple[str, ...]
    final_feature_probabilities: dict[str, float]
    aggregated_feature_probabilities: dict[str, float]
    seed: int
    steps: tuple[ClassPosteriorStep, ...]
    final_map_class: str
    final_weights: dict[str, float]
    aggregate_map_class: str
    aggregate_weights: dict[str, float]
    transient_map_class: str
    terminal_map_class: str


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    total_runs: int
    overall_accuracy: float
    transient_accuracy: float
    terminal_accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_counts: dict[str, dict[str, int]]
    transient_confusion_counts: dict[str, dict[str, int]]
    terminal_confusion_counts: dict[str, dict[str, int]]
    scenario_confusion_counts: dict[str, dict[str, int]]
    phase_confusion_counts: dict[str, dict[str, int]]
    scenario_phase_hit_counts: dict[str, dict[str, int]]
    scenario_phase_total_counts: dict[str, dict[str, int]]
    class_feature_detection_counts: dict[str, dict[str, int]]
    feature_confusion_counts: dict[str, dict[str, int]]
    true_feature_predicted_class_counts: dict[str, dict[str, int]]
    detected_feature_predicted_class_counts: dict[str, dict[str, int]]
    class_feature_precision: dict[str, dict[str, float]]
    class_feature_recall: dict[str, dict[str, float]]
    feature_class_lift: dict[str, dict[str, float]]
    unknown_retention_mean: float
    representative_traces: dict[str, tuple[dict[str, float], ...]]
    entropy_mean_by_step: tuple[float, ...]
    entropy_p90_by_step: tuple[float, ...]
    mean_feature_probability_by_step: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class ToyBenchmarkResult:
    config: ToyBenchmarkConfig
    dataset: ToyDataset
    runs: tuple[ClassificationRun, ...]
    summary: BenchmarkSummary
    representative_runs: dict[str, ClassificationRun]


def default_class_specs() -> tuple[ToyClassSpec, ...]:
    return (
        ToyClassSpec("coast", accel_bias=0.0, jerk_sigma=0.08, accel_limit=0.35, speed_limit=1.5, prior_weight=0.15),
        ToyClassSpec("drift", accel_bias=-0.02, jerk_sigma=0.08, accel_limit=0.35, speed_limit=1.8, prior_weight=0.15),
        ToyClassSpec("brake", accel_bias=-0.05, jerk_sigma=0.12, accel_limit=1.5, speed_limit=3.5, prior_weight=0.15),
        ToyClassSpec("maneuver", accel_bias=0.0, jerk_sigma=0.38, accel_limit=1.2, speed_limit=2.2, prior_weight=0.15),
        ToyClassSpec("powered", accel_bias=0.05, jerk_sigma=0.05, accel_limit=2.5, speed_limit=14.0, prior_weight=0.15),
        ToyClassSpec("unknown", accel_bias=0.0, jerk_sigma=1.15, accel_limit=8.0, speed_limit=8.0, prior_weight=0.1),
    )


def default_toy_scenarios() -> tuple[ToyScenarioSpec, ...]:
    return (
        ToyScenarioSpec("coast_nominal", "coast", velocity_range=(-0.12, 0.12), accel_range=(-0.05, 0.05), phase_targets=((0.0, 1.0, "steady"),), jerk_scale=0.9, accel_clip_scale=0.55, velocity_clip_scale=0.55),
        ToyScenarioSpec("coast_gusty", "coast", velocity_range=(-0.22, 0.22), accel_range=(-0.08, 0.08), phase_targets=((0.0, 0.70, "steady"), (0.70, 1.0, "oscillating")), jerk_scale=1.2, accel_clip_scale=0.70, velocity_clip_scale=0.65, obs_sigma_scale=1.1),
        ToyScenarioSpec("drift_nominal", "drift", velocity_range=(-1.45, -0.95), accel_range=(-0.06, 0.01), phase_targets=((0.0, 1.0, "reversing"),), jerk_scale=0.9, accel_clip_scale=0.30, velocity_clip_scale=0.92),
        ToyScenarioSpec("drift_fast_backtrack", "drift", velocity_range=(-1.75, -1.20), accel_range=(-0.10, -0.02), phase_targets=((0.0, 0.75, "reversing"), (0.75, 1.0, "edge")), jerk_scale=1.1, accel_clip_scale=0.35, velocity_clip_scale=0.96),
        ToyScenarioSpec("brake_soft_stop", "brake", velocity_range=(1.4, 2.4), accel_range=(-0.28, -0.10), phase_targets=((0.0, 0.55, "braking"), (0.55, 1.0, "steady")), jerk_scale=0.85, accel_clip_scale=0.88),
        ToyScenarioSpec("brake_hard_stop", "brake", velocity_range=(2.2, 3.3), accel_range=(-0.55, -0.20), phase_targets=((0.0, 0.65, "braking"), (0.65, 1.0, "edge")), jerk_scale=1.2, accel_clip_scale=0.98, obs_sigma_scale=1.05),
        ToyScenarioSpec("maneuver_weave", "maneuver", velocity_range=(-0.10, 0.50), accel_range=(-0.20, 0.20), phase_targets=((0.0, 1.0, "oscillating"),), jerk_scale=1.0, maneuver_impulse=0.58, flip_period_divisor=7),
        ToyScenarioSpec("maneuver_reversal", "maneuver", velocity_range=(-0.35, 0.75), accel_range=(-0.25, 0.25), phase_targets=((0.0, 0.40, "oscillating"), (0.40, 0.75, "reversing"), (0.75, 1.0, "oscillating")), jerk_scale=1.25, maneuver_impulse=0.75, flip_period_divisor=4, obs_sigma_scale=1.1),
        ToyScenarioSpec("powered_climb", "powered", velocity_range=(0.8, 1.7), accel_range=(0.15, 0.40), phase_targets=((0.0, 1.0, "powered"),), jerk_scale=0.9, accel_clip_scale=0.88),
        ToyScenarioSpec("powered_sprint", "powered", velocity_range=(1.2, 2.0), accel_range=(0.25, 0.55), phase_targets=((0.0, 0.70, "powered"), (0.70, 1.0, "edge")), jerk_scale=1.05, accel_clip_scale=0.95, obs_sigma_scale=1.05),
        ToyScenarioSpec("unknown_flip", "unknown", velocity_range=(-0.45, 0.45), accel_range=(-0.35, 0.35), phase_targets=((0.0, 0.34, "oscillating"), (0.34, 0.67, "powered"), (0.67, 1.0, "oscillating")), jerk_scale=0.9, unknown_bias=1.4, flip_period_divisor=6, obs_sigma_scale=1.1),
        ToyScenarioSpec("unknown_surge", "unknown", velocity_range=(-0.70, 0.70), accel_range=(-0.55, 0.55), phase_targets=((0.0, 0.34, "powered"), (0.34, 0.67, "reversing"), (0.67, 1.0, "edge")), jerk_scale=1.2, unknown_bias=1.8, flip_period_divisor=4, obs_sigma_scale=1.2),
    )


def _scenario_specs_by_class(config: ToyBenchmarkConfig) -> dict[str, tuple[ToyScenarioSpec, ...]]:
    grouped: dict[str, list[ToyScenarioSpec]] = {}
    for scenario in config.scenario_specs:
        grouped.setdefault(scenario.class_name, []).append(scenario)
    return {class_name: tuple(scenarios) for class_name, scenarios in grouped.items()}


def _initial_filter_state(
    spec: ToyClassSpec,
    first_observation: float,
    config: ToyBenchmarkConfig,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    if spec.name == "coast":
        mean = (first_observation, 0.0, 0.0)
        covariance = (
            (config.initial_position_variance, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.35),
        )
    elif spec.name == "drift":
        mean = (first_observation, -1.2, -0.05)
        covariance = (
            (config.initial_position_variance, 0.0, 0.0),
            (0.0, 0.8, 0.0),
            (0.0, 0.0, 0.2),
        )
    elif spec.name == "brake":
        mean = (first_observation, 2.2, -0.35)
        covariance = (
            (config.initial_position_variance, 0.0, 0.0),
            (0.0, 2.4, 0.0),
            (0.0, 0.0, 0.7),
        )
    elif spec.name == "powered":
        mean = (first_observation, 0.9, 0.25)
        covariance = (
            (config.initial_position_variance, 0.0, 0.0),
            (0.0, 3.5, 0.0),
            (0.0, 0.0, 0.7),
        )
    elif spec.name == "maneuver":
        mean = (first_observation, 0.2, 0.0)
        covariance = (
            (config.initial_position_variance, 0.0, 0.0),
            (0.0, 7.0, 0.0),
            (0.0, 0.0, 1.4),
        )
    else:
        mean = (first_observation, 0.0, 0.0)
        covariance = (
            (config.initial_position_variance, 0.0, 0.0),
            (0.0, 25.0, 0.0),
            (0.0, 0.0, 9.0),
        )
    return mean, covariance


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def _class_signature(name: str) -> tuple[float, float, float, float, float]:
    if name == "coast":
        return 0.0, 0.30, 0.0, 0.12, 0.15
    if name == "drift":
        return -1.25, 0.40, -0.03, 0.10, 0.10
    if name == "brake":
        return 2.20, 0.90, -0.32, 0.24, 0.10
    if name == "maneuver":
        return 0.30, 1.45, 0.0, 0.75, 0.85
    if name == "powered":
        return 1.35, 1.35, 0.30, 0.22, 0.10
    return 0.0, 3.80, 0.0, 1.60, 0.55


def _extract_true_features(track: ToyTrack, spec: ToyClassSpec) -> tuple[str, ...]:
    velocities = track.velocities_true
    accelerations = track.accelerations_true
    accel_sign_changes = sum(
        1
        for index in range(1, len(accelerations))
        if accelerations[index - 1] * accelerations[index] < -0.02
    )
    velocity_sign_changes = sum(
        1
        for index in range(1, len(velocities))
        if velocities[index - 1] * velocities[index] < -0.02
    )
    features = {
        "reverse_motion": min(velocities) < -0.75,
        "high_speed": max(abs(value) for value in velocities) > 1.75,
        "positive_accel": max(accelerations) > 0.25,
        "hard_brake": min(accelerations) < -0.25,
        "oscillatory": accel_sign_changes >= 2 or velocity_sign_changes >= 1,
        "near_envelope": any(
            abs(velocity) > 0.8 * spec.speed_limit or abs(accel) > 0.8 * spec.accel_limit
            for velocity, accel in zip(velocities, accelerations)
        ),
    }
    return tuple(feature for feature in FEATURE_NAMES if features[feature])


def _phase_label_from_kinematics(velocity: float, acceleration: float, prev_acceleration: float, speed_limit: float, accel_limit: float) -> str:
    if abs(velocity) > 0.8 * speed_limit or abs(acceleration) > 0.8 * accel_limit:
        return "edge"
    if prev_acceleration * acceleration < -0.02 or abs(acceleration - prev_acceleration) > 0.45 * max(accel_limit, 1e-6):
        return "oscillating"
    if velocity < -0.75:
        return "reversing"
    if acceleration < -0.18:
        return "braking"
    if acceleration > 0.18:
        return "powered"
    return "steady"


def _extract_true_phase_labels(track: ToyTrack, spec: ToyClassSpec) -> tuple[str, ...]:
    labels: list[str] = []
    prev_acceleration = track.accelerations_true[0] if track.accelerations_true else 0.0
    for velocity, acceleration in zip(track.velocities_true, track.accelerations_true):
        labels.append(
            _phase_label_from_kinematics(
                velocity,
                acceleration,
                prev_acceleration,
                spec.speed_limit,
                spec.accel_limit,
            )
        )
        prev_acceleration = acceleration
    return tuple(labels)


def _target_phase_labels_for_scenario(scenario_spec: ToyScenarioSpec, steps: int) -> tuple[str, ...]:
    if not scenario_spec.phase_targets:
        return tuple("steady" for _ in range(steps))
    labels: list[str] = []
    for step_index in range(steps):
        progress = step_index / max(steps - 1, 1)
        label = scenario_spec.phase_targets[-1][2]
        for start_fraction, end_fraction, candidate in scenario_spec.phase_targets:
            if start_fraction <= progress < end_fraction or (step_index == steps - 1 and progress <= end_fraction):
                label = candidate
                break
        labels.append(label)
    return tuple(labels)


def _make_track_from_scenario(
    class_spec: ToyClassSpec,
    scenario_spec: ToyScenarioSpec,
    steps: int,
    dt: float,
    seed: int,
    obs_sigma: float,
) -> ToyTrack:
    rng = random.Random(seed)
    position = rng.uniform(-1.0, 1.0)
    velocity = rng.uniform(*scenario_spec.velocity_range)
    acceleration = rng.uniform(*scenario_spec.accel_range)
    positions_true: list[float] = []
    velocities_true: list[float] = []
    accelerations_true: list[float] = []
    positions_obs: list[float] = []
    period = max(2, steps // max(2, scenario_spec.flip_period_divisor))
    direction = 1.0
    effective_obs_sigma = obs_sigma * scenario_spec.obs_sigma_scale

    for step in range(steps):
        if step > 0 and step % period == 0:
            direction *= -1.0

        positions_true.append(position)
        velocities_true.append(velocity)
        accelerations_true.append(acceleration)
        positions_obs.append(position + rng.gauss(0.0, effective_obs_sigma))

        next_position = position + velocity * dt + 0.5 * acceleration * dt * dt
        next_velocity = velocity + acceleration * dt
        next_acceleration = acceleration + class_spec.accel_bias * scenario_spec.accel_bias_scale * dt
        next_acceleration += rng.gauss(0.0, class_spec.jerk_sigma * scenario_spec.jerk_scale)

        if class_spec.name == "coast":
            next_acceleration = _clamp(next_acceleration, -scenario_spec.accel_clip_scale * class_spec.accel_limit, scenario_spec.accel_clip_scale * class_spec.accel_limit)
            next_velocity = _clamp(next_velocity, -scenario_spec.velocity_clip_scale * class_spec.speed_limit, scenario_spec.velocity_clip_scale * class_spec.speed_limit)
        elif class_spec.name == "drift":
            next_acceleration = _clamp(next_acceleration, -scenario_spec.accel_clip_scale * class_spec.accel_limit, 0.12 * class_spec.accel_limit)
            next_velocity = _clamp(next_velocity, -scenario_spec.velocity_clip_scale * class_spec.speed_limit, 0.1 * class_spec.speed_limit)
        elif class_spec.name == "brake":
            next_acceleration = _clamp(next_acceleration, -scenario_spec.accel_clip_scale * class_spec.accel_limit, -0.02)
            next_velocity = _clamp(next_velocity, 0.0, scenario_spec.velocity_clip_scale * class_spec.speed_limit)
        elif class_spec.name == "powered":
            next_acceleration = _clamp(next_acceleration, 0.08, scenario_spec.accel_clip_scale * class_spec.accel_limit)
            next_velocity = _clamp(next_velocity, 0.0, scenario_spec.velocity_clip_scale * class_spec.speed_limit)
        elif class_spec.name == "maneuver":
            next_acceleration = acceleration + direction * scenario_spec.maneuver_impulse
            next_acceleration += rng.gauss(0.0, class_spec.jerk_sigma * scenario_spec.jerk_scale)
            next_acceleration = _clamp(next_acceleration, -scenario_spec.accel_clip_scale * class_spec.accel_limit, scenario_spec.accel_clip_scale * class_spec.accel_limit)
            next_velocity = _clamp(next_velocity, -scenario_spec.velocity_clip_scale * class_spec.speed_limit, scenario_spec.velocity_clip_scale * class_spec.speed_limit)
        else:
            next_acceleration = acceleration + direction * scenario_spec.unknown_bias
            next_acceleration += rng.gauss(0.0, class_spec.jerk_sigma * scenario_spec.jerk_scale)
            next_acceleration = _clamp(next_acceleration, -scenario_spec.accel_clip_scale * class_spec.accel_limit, scenario_spec.accel_clip_scale * class_spec.accel_limit)
            next_velocity = _clamp(next_velocity, -scenario_spec.velocity_clip_scale * class_spec.speed_limit, scenario_spec.velocity_clip_scale * class_spec.speed_limit)

        position = next_position
        velocity = next_velocity
        acceleration = next_acceleration

    provisional_track = ToyTrack(
        class_name=class_spec.name,
        scenario_name=scenario_spec.name,
        true_features=(),
        true_phase_labels=(),
        target_phase_labels=(),
        seed=seed,
        dt=dt,
        positions_true=tuple(positions_true),
        velocities_true=tuple(velocities_true),
        accelerations_true=tuple(accelerations_true),
        positions_obs=tuple(positions_obs),
    )
    return ToyTrack(
        class_name=class_spec.name,
        scenario_name=scenario_spec.name,
        true_features=_extract_true_features(provisional_track, class_spec),
        true_phase_labels=_extract_true_phase_labels(provisional_track, class_spec),
        target_phase_labels=_target_phase_labels_for_scenario(scenario_spec, len(provisional_track.positions_true)),
        seed=seed,
        dt=dt,
        positions_true=provisional_track.positions_true,
        velocities_true=provisional_track.velocities_true,
        accelerations_true=provisional_track.accelerations_true,
        positions_obs=provisional_track.positions_obs,
    )


def generate_toy_track(spec: ToyClassSpec, steps: int, dt: float, seed: int, obs_sigma: float) -> ToyTrack:
    scenario_specs = [scenario for scenario in default_toy_scenarios() if scenario.class_name == spec.name]
    scenario = scenario_specs[seed % len(scenario_specs)]
    return _make_track_from_scenario(spec, scenario, steps, dt, seed, obs_sigma)


def generate_toy_dataset(config: ToyBenchmarkConfig, tracks_per_class: int, seed: int) -> ToyDataset:
    rng = random.Random(seed)
    scenario_specs_by_class = _scenario_specs_by_class(config)
    class_spec_by_name = {spec.name: spec for spec in config.class_specs}
    tracks: list[ToyTrack] = []
    for class_index, class_spec in enumerate(config.class_specs):
        scenarios = scenario_specs_by_class[class_spec.name]
        for track_index in range(tracks_per_class):
            scenario_spec = scenarios[track_index % len(scenarios)]
            track_seed = rng.randrange(1 << 30) + class_index * 10_000 + track_index
            tracks.append(
                _make_track_from_scenario(
                    class_spec=class_spec_by_name[scenario_spec.class_name],
                    scenario_spec=scenario_spec,
                    steps=config.steps,
                    dt=config.dt,
                    seed=track_seed,
                    obs_sigma=config.obs_sigma,
                )
            )
    return ToyDataset(
        tracks=tuple(tracks),
        class_specs=config.class_specs,
        scenario_specs=config.scenario_specs,
        steps=config.steps,
        dt=config.dt,
        obs_sigma=config.obs_sigma,
        seed=seed,
    )


def _predict_filter(
    mean: tuple[float, float, float],
    covariance: tuple[tuple[float, float, float], ...],
    dt: float,
    accel_bias: float,
    jerk_sigma: float,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    transition = _transition_matrix(dt)
    control = (0.0, 0.0, accel_bias * dt)
    process_noise = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, jerk_sigma * jerk_sigma),
    )
    predicted_mean = _add_vec3(_matvec3(transition, mean), control)
    predicted_covariance = _add_mat3(
        _matmul3(_matmul3(transition, covariance), _transpose3(transition)),
        process_noise,
    )
    return predicted_mean, predicted_covariance


def _innovation_log_likelihood(innovation: float, variance: float) -> float:
    safe_variance = max(variance, 1e-9)
    return -0.5 * (log(2.0 * pi * safe_variance) + (innovation * innovation) / safe_variance)


def _update_filter(
    predicted_mean: tuple[float, float, float],
    predicted_covariance: tuple[tuple[float, float, float], ...],
    measurement: float,
    observation_variance: float,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...], float, float]:
    innovation = measurement - predicted_mean[0]
    innovation_variance = max(predicted_covariance[0][0] + observation_variance, 1e-9)
    gain = tuple(predicted_covariance[row][0] / innovation_variance for row in range(3))
    updated_mean = tuple(predicted_mean[index] + gain[index] * innovation for index in range(3))

    updated_covariance = tuple(
        tuple(predicted_covariance[row][col] - gain[row] * predicted_covariance[0][col] for col in range(3))
        for row in range(3)
    )
    return updated_mean, updated_covariance, innovation, innovation_variance


def _feature_probabilities_from_state(
    class_specs: tuple[ToyClassSpec, ...],
    weights: dict[str, float],
    means: dict[str, tuple[float, float, float]],
    covariances: dict[str, tuple[tuple[float, float, float], ...]],
    previous_accel_means: dict[str, float],
) -> dict[str, float]:
    class_spec_by_name = {spec.name: spec for spec in class_specs}
    feature_probabilities = {feature: 0.0 for feature in FEATURE_NAMES}
    for class_name, weight in weights.items():
        spec = class_spec_by_name[class_name]
        mean = means[class_name]
        covariance = covariances[class_name]
        velocity_mean = mean[1]
        accel_mean = mean[2]
        velocity_variance = max(covariance[1][1], 1e-9)
        accel_variance = max(covariance[2][2], 1e-9)
        speed_inside = gaussian_interval_probability(velocity_mean, velocity_variance, 1.75)
        speed_limit_inside = gaussian_interval_probability(velocity_mean, velocity_variance, 0.8 * spec.speed_limit)
        accel_limit_inside = gaussian_interval_probability(accel_mean, accel_variance, 0.8 * spec.accel_limit)
        sign_flip = 1.0 if previous_accel_means[class_name] * accel_mean < -0.02 else 0.0
        oscillation_base = 0.15
        if class_name == "maneuver":
            oscillation_base = 0.55
        elif class_name == "unknown":
            oscillation_base = 0.45
        elif class_name in {"brake", "powered"}:
            oscillation_base = 0.25
        oscillatory_probability = _clamp(
            oscillation_base + 0.35 * sign_flip + 0.20 * min(1.0, abs(accel_mean) / max(spec.accel_limit, 1e-6)),
            0.0,
            1.0,
        )
        per_class_feature_probabilities = {
            "reverse_motion": _gaussian_lower_tail_probability(velocity_mean, velocity_variance, -0.75),
            "high_speed": 1.0 - speed_inside,
            "positive_accel": _gaussian_upper_tail_probability(accel_mean, accel_variance, 0.25),
            "hard_brake": _gaussian_lower_tail_probability(accel_mean, accel_variance, -0.18),
            "oscillatory": oscillatory_probability,
            "near_envelope": max(1.0 - speed_limit_inside, 1.0 - accel_limit_inside),
        }
        for feature_name, probability in per_class_feature_probabilities.items():
            feature_probabilities[feature_name] += weight * probability
    return {name: _clamp(value, 0.0, 1.0) for name, value in feature_probabilities.items()}


def _class_behavior_likelihood(
    spec: ToyClassSpec,
    updated_mean: tuple[float, float, float],
    updated_covariance: tuple[tuple[float, float, float], ...],
    previous_accel_mean: float,
    config: ToyBenchmarkConfig,
) -> tuple[float, dict[str, float]]:
    velocity_mean = updated_mean[1]
    accel_mean = updated_mean[2]
    velocity_variance = max(updated_covariance[1][1], 1e-9)
    accel_variance = max(updated_covariance[2][2], 1e-9)
    signature_velocity, velocity_sigma, signature_accel, accel_sigma, oscillation_target = _class_signature(spec.name)

    log_velocity_center = config.lambda_velocity_center * _gaussian_logpdf(
        velocity_mean,
        signature_velocity,
        velocity_variance + velocity_sigma * velocity_sigma,
    )
    log_accel_center = config.lambda_accel_center * _gaussian_logpdf(
        accel_mean,
        signature_accel,
        accel_variance + accel_sigma * accel_sigma,
    )

    near_zero_accel = gaussian_interval_probability(accel_mean, accel_variance, 0.12)
    positive_velocity = _gaussian_upper_tail_probability(velocity_mean, velocity_variance, 0.55)
    strong_positive_velocity = _gaussian_upper_tail_probability(velocity_mean, velocity_variance, 1.20)
    reverse_velocity = _gaussian_lower_tail_probability(velocity_mean, velocity_variance, -0.75)
    negative_accel = _gaussian_lower_tail_probability(accel_mean, accel_variance, -0.12)
    positive_accel = _gaussian_upper_tail_probability(accel_mean, accel_variance, 0.12)

    direction_score = 0.5
    if spec.name == "drift":
        direction_score = _clamp(0.75 * reverse_velocity + 0.25 * near_zero_accel, 0.0, 1.0)
    elif spec.name == "brake":
        direction_score = _clamp(0.45 * positive_velocity + 0.35 * negative_accel + 0.20 * strong_positive_velocity, 0.0, 1.0)
    elif spec.name == "powered":
        direction_score = _clamp(0.45 * positive_velocity + 0.35 * positive_accel + 0.20 * strong_positive_velocity, 0.0, 1.0)
    elif spec.name == "coast":
        direction_score = _clamp(0.60 * gaussian_interval_probability(velocity_mean, velocity_variance, 0.45) + 0.40 * near_zero_accel, 0.0, 1.0)
    elif spec.name == "maneuver":
        direction_score = _clamp(0.35 * gaussian_interval_probability(velocity_mean, velocity_variance, spec.speed_limit * 0.80) + 0.65 * max(negative_accel, positive_accel), 0.0, 1.0)
    elif spec.name == "unknown":
        direction_score = gaussian_interval_probability(velocity_mean, velocity_variance, 2.8)
    log_direction = config.lambda_direction_match * log(max(direction_score, config.epsilon))

    accel_flip_score = 1.0 if previous_accel_mean * accel_mean < -0.02 else 0.0
    normalized_accel = min(1.0, abs(accel_mean) / max(spec.accel_limit, 1e-6))
    oscillation_score = _clamp(0.20 + 0.55 * accel_flip_score + 0.25 * normalized_accel, 0.0, 1.0)
    if spec.name == "coast":
        oscillation_likelihood = 1.0 - oscillation_score
    elif spec.name == "drift":
        oscillation_likelihood = _clamp(0.85 - 0.60 * oscillation_score, 0.0, 1.0)
    elif spec.name == "brake":
        oscillation_likelihood = _clamp(0.82 - 0.30 * oscillation_score, 0.0, 1.0)
    elif spec.name == "powered":
        oscillation_likelihood = _clamp(0.78 - 0.50 * oscillation_score, 0.0, 1.0)
    elif spec.name == "maneuver":
        oscillation_likelihood = _clamp(0.25 + 0.75 * oscillation_score, 0.0, 1.0)
    else:
        oscillation_likelihood = _clamp(0.25 + 0.45 * (1.0 - abs(oscillation_score - oscillation_target)), 0.0, 1.0)
    log_oscillation = config.lambda_oscillation_match * log(max(oscillation_likelihood, config.epsilon))

    total = log_velocity_center + log_accel_center + log_direction + log_oscillation
    return total, {
        "velocity_center": log_velocity_center,
        "accel_center": log_accel_center,
        "direction": log_direction,
        "oscillation": log_oscillation,
    }


def _observed_kinematics_likelihood(
    spec: ToyClassSpec,
    recent_measurements: tuple[float, ...],
    dt: float,
    observation_variance: float,
    config: ToyBenchmarkConfig,
) -> tuple[float, dict[str, float]]:
    if len(recent_measurements) < 3:
        return 0.0, {"obs_velocity": 0.0, "obs_accel": 0.0}

    observed_velocity = (recent_measurements[-1] - recent_measurements[-2]) / dt
    observed_accel = (recent_measurements[-1] - 2.0 * recent_measurements[-2] + recent_measurements[-3]) / (dt * dt)
    signature_velocity, velocity_sigma, signature_accel, accel_sigma, _ = _class_signature(spec.name)
    observed_velocity_variance = max((2.0 * observation_variance) / (dt * dt), 1e-9)
    observed_accel_variance = max((6.0 * observation_variance) / (dt * dt * dt * dt), 1e-9)
    log_obs_velocity = config.lambda_observed_velocity * _gaussian_logpdf(
        observed_velocity,
        signature_velocity,
        observed_velocity_variance + velocity_sigma * velocity_sigma,
    )
    log_obs_accel = config.lambda_observed_accel * _gaussian_logpdf(
        observed_accel,
        signature_accel,
        observed_accel_variance + accel_sigma * accel_sigma,
    )
    return log_obs_velocity + log_obs_accel, {
        "obs_velocity": log_obs_velocity,
        "obs_accel": log_obs_accel,
    }


def _class_mode_log_likelihood(
    spec: ToyClassSpec,
    updated_mean: tuple[float, float, float],
    recent_measurements: tuple[float, ...],
    dt: float,
    step_index: int,
    total_steps: int,
) -> tuple[float, dict[str, float]]:
    if spec.name not in {"brake", "maneuver"}:
        return 0.0, {"mode_mix": 0.0}

    velocity = updated_mean[1]
    acceleration = updated_mean[2]
    if len(recent_measurements) >= 3:
        obs_velocity = (recent_measurements[-1] - recent_measurements[-2]) / dt
        obs_accel = (recent_measurements[-1] - 2.0 * recent_measurements[-2] + recent_measurements[-3]) / (dt * dt)
    else:
        obs_velocity = velocity
        obs_accel = acceleration

    progress = (step_index + 1) / max(total_steps, 1)
    mode_logs: list[float] = []
    if spec.name == "brake":
        early_weight = max(0.20, 1.10 - 1.35 * progress)
        mid_weight = 0.55 + 0.45 * (1.0 - abs(progress - 0.45) / 0.45)
        late_weight = max(0.18, 0.15 + 1.05 * progress)
        entry = (
            0.35 * _gaussian_logpdf(max(velocity, 0.0), 1.8, 1.05)
            + 0.35 * _gaussian_logpdf(acceleration, -0.28, 0.24)
            + 0.30 * _gaussian_logpdf(obs_accel, -0.24, 0.30)
            + log(max(early_weight, 1e-9))
        )
        rolling = (
            0.40 * _gaussian_logpdf(max(obs_velocity, 0.0), 0.85, 0.75)
            + 0.35 * _gaussian_logpdf(acceleration, -0.16, 0.18)
            + 0.25 * _gaussian_logpdf(obs_accel, -0.14, 0.22)
            + log(max(mid_weight, 1e-9))
        )
        terminal = (
            0.45 * _gaussian_logpdf(max(velocity, 0.0), 0.22, 0.30)
            + 0.25 * _gaussian_logpdf(max(obs_velocity, 0.0), 0.18, 0.28)
            + 0.30 * _gaussian_logpdf(acceleration, -0.06, 0.10)
            + log(max(late_weight, 1e-9))
        )
        mode_logs = [entry, rolling, terminal]
    else:
        reversal_energy = min(1.0, abs(obs_velocity) / 2.2)
        accel_drive = min(1.0, abs(obs_accel) / 1.1)
        weave = (
            0.25 * _gaussian_logpdf(acceleration, 0.0, 0.85)
            + 0.20 * _gaussian_logpdf(obs_accel, 0.0, 1.05)
            + 0.30 * log(max(0.25 + accel_drive, 1e-9))
            + 0.25 * log(max(0.20 + reversal_energy, 1e-9))
        )
        reversal = (
            0.20 * _gaussian_logpdf(velocity, 0.0, 2.40)
            + 0.20 * _gaussian_logpdf(obs_velocity, 0.0, 2.40)
            + 0.30 * log(max(0.15 + reversal_energy, 1e-9))
            + 0.30 * log(max(0.15 + accel_drive, 1e-9))
        )
        settle = (
            0.35 * _gaussian_logpdf(velocity, 0.20, 1.10)
            + 0.20 * _gaussian_logpdf(acceleration, 0.0, 0.55)
            + 0.20 * log(max(progress + 0.10, 1e-9))
            + 0.25 * log(max(0.20 + reversal_energy, 1e-9))
        )
        mode_logs = [weave, reversal, settle]

    raw_log_mix = _logsumexp(mode_logs) - log(len(mode_logs))
    if spec.name == "brake":
        log_mix = 1.60 + 0.55 * raw_log_mix
    else:
        log_mix = 1.05 + 0.70 * raw_log_mix
    return log_mix, {"mode_mix": log_mix}


def _detected_phase_label(
    class_specs: tuple[ToyClassSpec, ...],
    weights: dict[str, float],
    means: dict[str, tuple[float, float, float]],
    previous_accel_means: dict[str, float],
    feature_probabilities: dict[str, float],
) -> str:
    speed_limit = sum(weights[name] * spec.speed_limit for name, spec in ((spec.name, spec) for spec in class_specs))
    accel_limit = sum(weights[name] * spec.accel_limit for name, spec in ((spec.name, spec) for spec in class_specs))
    blended_velocity = sum(weights[name] * means[name][1] for name in weights)
    blended_acceleration = sum(weights[name] * means[name][2] for name in weights)
    blended_prev_acceleration = sum(weights[name] * previous_accel_means[name] for name in weights)
    phase = _phase_label_from_kinematics(
        blended_velocity,
        blended_acceleration,
        blended_prev_acceleration,
        speed_limit if speed_limit > 0 else 1.0,
        accel_limit if accel_limit > 0 else 1.0,
    )
    if feature_probabilities["oscillatory"] >= 0.55:
        return "oscillating"
    if feature_probabilities["near_envelope"] >= 0.50:
        return "edge"
    return phase


def run_class_bank(
    track: ToyTrack,
    class_specs: tuple[ToyClassSpec, ...],
    config: ToyBenchmarkConfig,
) -> ClassificationRun:
    initial_weights = _normalize_weights({spec.name: spec.prior_weight for spec in class_specs})
    initial_states = {
        spec.name: _initial_filter_state(spec, track.positions_obs[0], config)
        for spec in class_specs
    }
    means = {spec.name: initial_states[spec.name][0] for spec in class_specs}
    covariances = {spec.name: initial_states[spec.name][1] for spec in class_specs}
    previous_accel_means = {spec.name: means[spec.name][2] for spec in class_specs}

    weights = dict(initial_weights)
    steps: list[ClassPosteriorStep] = []
    observation_variance = config.obs_sigma * config.obs_sigma
    measurement_history = [track.positions_obs[0]]

    for measurement in track.positions_obs[1:]:
        measurement_history.append(measurement)
        predicted_weights = dict(weights)
        log_weight_updates: dict[str, float] = {}
        log_likelihood_terms: dict[str, dict[str, float]] = {}
        previous_accel_snapshot = dict(previous_accel_means)

        for spec in class_specs:
            predicted_mean, predicted_covariance = _predict_filter(
                mean=means[spec.name],
                covariance=covariances[spec.name],
                dt=track.dt,
                accel_bias=spec.accel_bias,
                jerk_sigma=spec.jerk_sigma,
            )
            updated_mean, updated_covariance, innovation, innovation_variance = _update_filter(
                predicted_mean=predicted_mean,
                predicted_covariance=predicted_covariance,
                measurement=measurement,
                observation_variance=observation_variance,
            )

            means[spec.name] = updated_mean
            covariances[spec.name] = updated_covariance

            log_dyn = _innovation_log_likelihood(innovation, innovation_variance)
            speed_limit = spec.speed_limit
            if spec.name == "maneuver":
                # Keep maneuver broad enough for reversals, but do not
                # inflate the envelope so far that steady reverse drift
                # looks equally plausible as an oscillatory maneuver.
                speed_limit = 0.95 * spec.speed_limit
            speed_probability = gaussian_interval_probability(
                mean=updated_mean[1],
                variance=updated_covariance[1][1],
                limit=speed_limit,
            )
            accel_probability = gaussian_interval_probability(
                mean=updated_mean[2],
                variance=updated_covariance[2][2],
                limit=spec.accel_limit,
            )
            log_speed = config.lambda_v * log(config.epsilon + speed_probability)
            log_accel = config.lambda_a * log(config.epsilon + accel_probability)
            log_behavior, behavior_terms = _class_behavior_likelihood(
                spec=spec,
                updated_mean=updated_mean,
                updated_covariance=updated_covariance,
                previous_accel_mean=previous_accel_means[spec.name],
                config=config,
            )
            log_observed, observed_terms = _observed_kinematics_likelihood(
                spec=spec,
                recent_measurements=tuple(measurement_history[-3:]),
                dt=track.dt,
                observation_variance=observation_variance,
                config=config,
            )
            log_mode_mix, mode_terms = _class_mode_log_likelihood(
                spec=spec,
                updated_mean=updated_mean,
                recent_measurements=tuple(measurement_history[-3:]),
                dt=track.dt,
                step_index=len(steps),
                total_steps=len(track.positions_obs) - 1,
            )
            log_total = log_dyn + log_speed + log_accel + log_behavior + log_observed + log_mode_mix
            if spec.name == "unknown":
                log_total -= config.unknown_log_penalty

            log_weight_updates[spec.name] = log(max(predicted_weights[spec.name], config.epsilon)) + log_total
            log_likelihood_terms[spec.name] = {
                "dyn": log_dyn,
                "speed": log_speed,
                "accel": log_accel,
                **behavior_terms,
                **observed_terms,
                **mode_terms,
                "unknown_penalty": (-config.unknown_log_penalty if spec.name == "unknown" else 0.0),
                "total": log_total,
            }

        log_norm = _logsumexp(list(log_weight_updates.values()))
        weights = {name: exp(log_weight_updates[name] - log_norm) for name in log_weight_updates}
        feature_probabilities = _feature_probabilities_from_state(
            class_specs=class_specs,
            weights=weights,
            means=means,
            covariances=covariances,
            previous_accel_means=previous_accel_means,
        )
        detected_phase_label = _detected_phase_label(
            class_specs=class_specs,
            weights=weights,
            means=means,
            previous_accel_means=previous_accel_snapshot,
            feature_probabilities=feature_probabilities,
        )
        for class_name in previous_accel_means:
            previous_accel_means[class_name] = means[class_name][2]
        map_class = max(weights, key=weights.get)
        steps.append(
            ClassPosteriorStep(
                predicted_class_weights=predicted_weights,
                updated_class_weights=dict(weights),
                log_likelihood_terms=log_likelihood_terms,
                feature_probabilities=feature_probabilities,
                detected_phase_label=detected_phase_label,
                posterior_entropy=_entropy(weights),
                map_class=map_class,
            )
        )

    final_weights = dict(weights)
    final_feature_probabilities = dict(steps[-1].feature_probabilities if steps else {feature: 0.0 for feature in FEATURE_NAMES})
    aggregated_feature_probabilities = _aggregate_feature_probabilities(tuple(steps))
    detected_features = tuple(
        feature_name
        for feature_name in FEATURE_NAMES
        if aggregated_feature_probabilities.get(feature_name, 0.0) >= _feature_detection_threshold(feature_name, config)
    )
    final_map_class = max(final_weights, key=final_weights.get)
    if steps:
        aggregate_weights = {
            spec.name: sum(step.updated_class_weights[spec.name] for step in steps) / len(steps)
            for spec in class_specs
        }
    else:
        aggregate_weights = dict(final_weights)
    aggregate_map_class = max(aggregate_weights, key=aggregate_weights.get)
    if steps:
        transient_end = max(1, len(steps) // 3)
        terminal_start = max(0, len(steps) - max(1, len(steps) // 3))
        transient_weights = {
            spec.name: sum(step.updated_class_weights[spec.name] for step in steps[:transient_end]) / transient_end
            for spec in class_specs
        }
        terminal_slice = steps[terminal_start:]
        terminal_weights = {
            spec.name: sum(step.updated_class_weights[spec.name] for step in terminal_slice) / len(terminal_slice)
            for spec in class_specs
        }
    else:
        transient_weights = dict(final_weights)
        terminal_weights = dict(final_weights)
    transient_map_class = max(transient_weights, key=transient_weights.get)
    terminal_map_class = max(terminal_weights, key=terminal_weights.get)
    return ClassificationRun(
        true_class=track.class_name,
        scenario_name=track.scenario_name,
        true_features=track.true_features,
        true_phase_labels=track.true_phase_labels,
        target_phase_labels=track.target_phase_labels,
        detected_features=detected_features,
        final_feature_probabilities=final_feature_probabilities,
        aggregated_feature_probabilities=aggregated_feature_probabilities,
        seed=track.seed,
        steps=tuple(steps),
        final_map_class=final_map_class,
        final_weights=final_weights,
        aggregate_map_class=aggregate_map_class,
        aggregate_weights=aggregate_weights,
        transient_map_class=transient_map_class,
        terminal_map_class=terminal_map_class,
    )


def summarize_runs(
    runs: tuple[ClassificationRun, ...],
    representative_trace_steps: int = 5,
) -> BenchmarkSummary:
    true_classes = sorted({run.true_class for run in runs})
    predicted_classes = sorted({run.aggregate_map_class for run in runs} | set(true_classes))
    confusion_counts = {
        true_class: {predicted_class: 0 for predicted_class in predicted_classes}
        for true_class in true_classes
    }
    transient_confusion_counts = {
        true_class: {predicted_class: 0 for predicted_class in predicted_classes}
        for true_class in true_classes
    }
    terminal_confusion_counts = {
        true_class: {predicted_class: 0 for predicted_class in predicted_classes}
        for true_class in true_classes
    }
    scenario_names = sorted({run.scenario_name for run in runs})
    scenario_confusion_counts = {
        scenario_name: {predicted_class: 0 for predicted_class in predicted_classes}
        for scenario_name in scenario_names
    }
    phase_confusion_counts = {
        phase_name: {detected_phase: 0 for detected_phase in PHASE_NAMES}
        for phase_name in PHASE_NAMES
    }
    scenario_phase_hit_counts = {
        scenario_name: {phase_name: 0 for phase_name in PHASE_NAMES}
        for scenario_name in scenario_names
    }
    scenario_phase_total_counts = {
        scenario_name: {phase_name: 0 for phase_name in PHASE_NAMES}
        for scenario_name in scenario_names
    }
    class_feature_detection_counts = {
        true_class: {feature_name: 0 for feature_name in FEATURE_NAMES}
        for true_class in true_classes
    }
    class_feature_truth_counts = {
        true_class: {feature_name: 0 for feature_name in FEATURE_NAMES}
        for true_class in true_classes
    }
    class_feature_true_positive_counts = {
        true_class: {feature_name: 0 for feature_name in FEATURE_NAMES}
        for true_class in true_classes
    }
    feature_confusion_counts = {
        feature_name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for feature_name in FEATURE_NAMES
    }
    true_feature_predicted_class_counts = {
        feature_name: {predicted_class: 0 for predicted_class in predicted_classes}
        for feature_name in FEATURE_NAMES
    }
    detected_feature_predicted_class_counts = {
        feature_name: {predicted_class: 0 for predicted_class in predicted_classes}
        for feature_name in FEATURE_NAMES
    }
    feature_detected_totals = {feature_name: 0 for feature_name in FEATURE_NAMES}
    feature_truth_totals = {feature_name: 0 for feature_name in FEATURE_NAMES}
    per_class_totals = {true_class: 0 for true_class in true_classes}
    per_class_correct = {true_class: 0 for true_class in true_classes}
    transient_correct = 0
    terminal_correct = 0
    unknown_weights: list[float] = []
    representative_traces: dict[str, tuple[dict[str, float], ...]] = {}

    step_count = max((len(run.steps) for run in runs), default=0)
    entropies_by_step: list[list[float]] = [[] for _ in range(step_count)]
    feature_probabilities_by_step = {feature_name: [[] for _ in range(step_count)] for feature_name in FEATURE_NAMES}

    for run in runs:
        confusion_counts[run.true_class][run.aggregate_map_class] += 1
        transient_confusion_counts[run.true_class][run.transient_map_class] += 1
        terminal_confusion_counts[run.true_class][run.terminal_map_class] += 1
        scenario_confusion_counts[run.scenario_name][run.aggregate_map_class] += 1
        per_class_totals[run.true_class] += 1
        if run.true_class == run.aggregate_map_class:
            per_class_correct[run.true_class] += 1
        if run.true_class == run.transient_map_class:
            transient_correct += 1
        if run.true_class == run.terminal_map_class:
            terminal_correct += 1
        if "unknown" in run.final_weights:
            unknown_weights.append(run.final_weights["unknown"])
        if run.true_class not in representative_traces:
            trace = tuple(step.updated_class_weights for step in run.steps[-representative_trace_steps:])
            representative_traces[run.true_class] = trace
        for feature_name in run.detected_features:
            class_feature_detection_counts[run.true_class][feature_name] += 1
        true_feature_set = set(run.true_features)
        detected_feature_set = set(run.detected_features)
        for feature_name in FEATURE_NAMES:
            if feature_name in true_feature_set:
                class_feature_truth_counts[run.true_class][feature_name] += 1
                feature_truth_totals[feature_name] += 1
                true_feature_predicted_class_counts[feature_name][run.aggregate_map_class] += 1
            if feature_name in detected_feature_set:
                feature_detected_totals[feature_name] += 1
                detected_feature_predicted_class_counts[feature_name][run.aggregate_map_class] += 1
            in_truth = feature_name in true_feature_set
            in_detected = feature_name in detected_feature_set
            if in_truth and in_detected:
                class_feature_true_positive_counts[run.true_class][feature_name] += 1
                feature_confusion_counts[feature_name]["tp"] += 1
            elif in_truth:
                feature_confusion_counts[feature_name]["fn"] += 1
            elif in_detected:
                feature_confusion_counts[feature_name]["fp"] += 1
            else:
                feature_confusion_counts[feature_name]["tn"] += 1
        for step_index, step in enumerate(run.steps):
            entropies_by_step[step_index].append(step.posterior_entropy)
            true_phase_index = min(step_index + 1, len(run.true_phase_labels) - 1) if run.true_phase_labels else 0
            if run.true_phase_labels:
                true_phase = run.true_phase_labels[true_phase_index]
                phase_confusion_counts[true_phase][step.detected_phase_label] += 1
            target_phase_index = min(step_index + 1, len(run.target_phase_labels) - 1) if run.target_phase_labels else 0
            if run.target_phase_labels:
                target_phase = run.target_phase_labels[target_phase_index]
                scenario_phase_total_counts[run.scenario_name][target_phase] += 1
                if step.detected_phase_label == target_phase:
                    scenario_phase_hit_counts[run.scenario_name][target_phase] += 1
            for feature_name in FEATURE_NAMES:
                feature_probabilities_by_step[feature_name][step_index].append(step.feature_probabilities[feature_name])

    total_runs = len(runs)
    total_correct = sum(per_class_correct.values())
    per_class_accuracy = {
        class_name: (per_class_correct[class_name] / per_class_totals[class_name])
        for class_name in true_classes
    }
    entropy_mean_by_step = tuple(
        sum(values) / len(values) if values else 0.0
        for values in entropies_by_step
    )
    entropy_p90_by_step = tuple(_quantile(values, 0.90) for values in entropies_by_step)
    mean_feature_probability_by_step = {
        feature_name: tuple(
            sum(values) / len(values) if values else 0.0
            for values in per_step_values
        )
        for feature_name, per_step_values in feature_probabilities_by_step.items()
    }
    class_feature_precision = {
        class_name: {
            feature_name: (
                class_feature_detection_counts[class_name][feature_name] / feature_detected_totals[feature_name]
                if feature_detected_totals[feature_name]
                else 0.0
            )
            for feature_name in FEATURE_NAMES
        }
        for class_name in true_classes
    }
    class_feature_recall = {
        class_name: {
            feature_name: (
                class_feature_true_positive_counts[class_name][feature_name] / class_feature_truth_counts[class_name][feature_name]
                if class_feature_truth_counts[class_name][feature_name]
                else 0.0
            )
            for feature_name in FEATURE_NAMES
        }
        for class_name in true_classes
    }
    feature_class_lift = {
        feature_name: {
            class_name: (
                (class_feature_detection_counts[class_name][feature_name] / per_class_totals[class_name])
                / (feature_detected_totals[feature_name] / total_runs)
                if per_class_totals[class_name] and feature_detected_totals[feature_name] and total_runs
                else 0.0
            )
            for class_name in true_classes
        }
        for feature_name in FEATURE_NAMES
    }
    return BenchmarkSummary(
        total_runs=total_runs,
        overall_accuracy=total_correct / total_runs if total_runs else 0.0,
        transient_accuracy=transient_correct / total_runs if total_runs else 0.0,
        terminal_accuracy=terminal_correct / total_runs if total_runs else 0.0,
        per_class_accuracy=per_class_accuracy,
        confusion_counts=confusion_counts,
        transient_confusion_counts=transient_confusion_counts,
        terminal_confusion_counts=terminal_confusion_counts,
        scenario_confusion_counts=scenario_confusion_counts,
        phase_confusion_counts=phase_confusion_counts,
        scenario_phase_hit_counts=scenario_phase_hit_counts,
        scenario_phase_total_counts=scenario_phase_total_counts,
        class_feature_detection_counts=class_feature_detection_counts,
        feature_confusion_counts=feature_confusion_counts,
        true_feature_predicted_class_counts=true_feature_predicted_class_counts,
        detected_feature_predicted_class_counts=detected_feature_predicted_class_counts,
        class_feature_precision=class_feature_precision,
        class_feature_recall=class_feature_recall,
        feature_class_lift=feature_class_lift,
        unknown_retention_mean=(sum(unknown_weights) / len(unknown_weights) if unknown_weights else 0.0),
        representative_traces=representative_traces,
        entropy_mean_by_step=entropy_mean_by_step,
        entropy_p90_by_step=entropy_p90_by_step,
        mean_feature_probability_by_step=mean_feature_probability_by_step,
    )


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/kinematic-classifier-sandbox-mpl")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _top_feature_list(values: dict[str, float], *, minimum: float = 0.0, limit: int = 3) -> list[tuple[str, float]]:
    ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    return [(name, value) for name, value in ranked if value > minimum][:limit]


def _aggregate_feature_probabilities(steps: tuple[ClassPosteriorStep, ...]) -> dict[str, float]:
    if not steps:
        return {feature_name: 0.0 for feature_name in FEATURE_NAMES}
    aggregated: dict[str, float] = {}
    for feature_name in FEATURE_NAMES:
        values = [step.feature_probabilities[feature_name] for step in steps]
        if feature_name in {"hard_brake", "oscillatory", "near_envelope"}:
            aggregated[feature_name] = max(values)
        elif feature_name in {"high_speed", "positive_accel", "reverse_motion"}:
            aggregated[feature_name] = max(sum(values) / len(values), max(values) * 0.85)
        else:
            aggregated[feature_name] = sum(values) / len(values)
    return aggregated


def _feature_detection_threshold(feature_name: str, config: ToyBenchmarkConfig) -> float:
    if feature_name == "hard_brake":
        return max(config.feature_detection_threshold, 0.72)
    if feature_name == "oscillatory":
        return max(config.feature_detection_threshold, 0.68)
    if feature_name == "near_envelope":
        return max(config.feature_detection_threshold, 0.70)
    if feature_name == "positive_accel":
        return max(config.feature_detection_threshold, 0.60)
    return config.feature_detection_threshold


def _plot_toy_benchmark(result: ToyBenchmarkResult):
    plt = _prepare_matplotlib()
    class_names = tuple(spec.name for spec in result.dataset.class_specs)
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(index % cmap.N) for index, name in enumerate(class_names)}
    runs = [result.representative_runs[name] for name in class_names if name in result.representative_runs]
    count = len(runs)
    cols = 2 if count > 1 else 1
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows), sharex=True, sharey=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax in axes_list[count:]:
        ax.set_visible(False)

    for ax, (class_name, run) in zip(axes_list, [(name, result.representative_runs[name]) for name in class_names if name in result.representative_runs]):
        steps = list(range(len(run.steps)))
        for series_name in class_names:
            ax.plot(
                steps,
                [step.updated_class_weights[series_name] for step in run.steps],
                label=series_name,
                color=colors[series_name],
                linewidth=2.4 if series_name == class_name else 1.4,
                alpha=0.95 if series_name == class_name else 0.8,
            )
        ax.set_title(f"{class_name} posterior", loc="left", fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, max(len(run.steps) - 1, 1))
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Step")
        ax.set_ylabel("Posterior")
        ax.text(
            0.98,
            0.06,
            f"MAP: {run.final_map_class}\nscenario: {run.scenario_name}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    if axes_list:
        axes_list[0].legend(loc="upper center", ncol=min(4, len(class_names)), bbox_to_anchor=(1.0, 1.24), frameon=False)
    fig.suptitle("1D Toy Bayesian Benchmark Posterior Evolution", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_toy_benchmark_svg(result: ToyBenchmarkResult) -> str:
    fig = _plot_toy_benchmark(result)
    try:
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        fig.clf()


def render_toy_benchmark_png_bytes(result: ToyBenchmarkResult) -> bytes:
    fig = _plot_toy_benchmark(result)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        fig.clf()


def _build_toy_feature_confusion_figure(result: ToyBenchmarkResult):
    plt = _prepare_matplotlib()
    summary = result.summary
    class_names = sorted(summary.confusion_counts)
    feature_names = list(FEATURE_NAMES)
    true_matrix = [
        [summary.true_feature_predicted_class_counts[feature_name][class_name] for class_name in class_names]
        for feature_name in feature_names
    ]
    detected_matrix = [
        [summary.detected_feature_predicted_class_counts[feature_name][class_name] for class_name in class_names]
        for feature_name in feature_names
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.9 * len(feature_names) + 3.5))
    true_ax, detected_ax = axes

    for ax, matrix, title in (
        (true_ax, true_matrix, "True Feature vs Predicted Class"),
        (detected_ax, detected_matrix, "Detected Feature vs Predicted Class"),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(feature_names)), feature_names)
        matrix_max = max((max(row) for row in matrix), default=0)
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                text_color = "#ffffff" if value > max(1, matrix_max * 0.45) else "#111827"
                ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8, color=text_color)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Toy 1D Feature-Class Confusion Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def render_toy_feature_confusion_svg(result: ToyBenchmarkResult) -> str:
    fig = _build_toy_feature_confusion_figure(result)
    try:
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        fig.clf()


def render_toy_feature_confusion_png_bytes(result: ToyBenchmarkResult) -> bytes:
    fig = _build_toy_feature_confusion_figure(result)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        fig.clf()


def write_toy_benchmark_trace_csv(result: ToyBenchmarkResult, output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "toy_1d_benchmark_traces.csv"
    class_names = tuple(spec.name for spec in result.dataset.class_specs)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_index",
                "scenario_name",
                "true_class",
                "seed",
                "step",
                "true_phase_label",
                "target_phase_label",
                "detected_phase_label",
                "map_class",
                "aggregate_map_class",
                "transient_map_class",
                "terminal_map_class",
                "posterior_entropy",
                *class_names,
                *FEATURE_NAMES,
                *[f"agg_{feature_name}" for feature_name in FEATURE_NAMES],
            ]
        )
        for run_index, run in enumerate(result.runs):
            for step_idx, step in enumerate(run.steps):
                writer.writerow(
                    [
                        run_index,
                        run.scenario_name,
                        run.true_class,
                        run.seed,
                        step_idx,
                        run.true_phase_labels[min(step_idx + 1, len(run.true_phase_labels) - 1)] if run.true_phase_labels else "",
                        run.target_phase_labels[min(step_idx + 1, len(run.target_phase_labels) - 1)] if run.target_phase_labels else "",
                        step.detected_phase_label,
                        step.map_class,
                        run.aggregate_map_class,
                        run.transient_map_class,
                        run.terminal_map_class,
                        f"{step.posterior_entropy:.6f}",
                        *(f"{step.updated_class_weights[name]:.6f}" for name in class_names),
                        *(f"{step.feature_probabilities[feature_name]:.6f}" for feature_name in FEATURE_NAMES),
                        *(f"{run.aggregated_feature_probabilities[feature_name]:.6f}" for feature_name in FEATURE_NAMES),
                    ]
                )
    return output_path


def render_toy_benchmark_markdown(result: ToyBenchmarkResult) -> str:
    summary = result.summary
    scenario_counts: dict[str, int] = {}
    for track in result.dataset.tracks:
        scenario_counts[track.scenario_name] = scenario_counts.get(track.scenario_name, 0) + 1
    lines = [
        "# 1D Toy Bayesian Benchmark Report",
        "",
        f"Seed: {result.dataset.seed}",
        f"Steps: {result.dataset.steps}",
        f"Tracks per class: {len(result.dataset.tracks) // len(result.dataset.class_specs)}",
        f"Observation sigma: {result.dataset.obs_sigma:.3f}",
        f"Overall accuracy: {summary.overall_accuracy:.3f}",
        f"Transient accuracy: {summary.transient_accuracy:.3f}",
        f"Terminal accuracy: {summary.terminal_accuracy:.3f}",
        f"Unknown retention mean: {summary.unknown_retention_mean:.3f}",
        "",
        "## Monte Carlo Scenario Coverage",
        "",
    ]
    for scenario_name in sorted(scenario_counts):
        lines.append(f"- {scenario_name}: {scenario_counts[scenario_name]} runs")

    lines.extend(["", "## Per-Class Accuracy", ""])
    for class_name in sorted(summary.per_class_accuracy):
        lines.append(f"- {class_name}: {summary.per_class_accuracy[class_name]:.3f}")

    lines.extend(["", "## Class Confusion Counts", "", "These counts use aggregate trajectory posteriors rather than only the last step.", ""])
    for true_class in sorted(summary.confusion_counts):
        counts = ", ".join(
            f"{pred}={summary.confusion_counts[true_class][pred]}"
            for pred in sorted(summary.confusion_counts[true_class])
        )
        lines.append(f"- {true_class}: {counts}")

    lines.extend(["", "## Transient Class Confusion Counts", "", "These counts average the first third of each trajectory.", ""])
    for true_class in sorted(summary.transient_confusion_counts):
        counts = ", ".join(
            f"{pred}={summary.transient_confusion_counts[true_class][pred]}"
            for pred in sorted(summary.transient_confusion_counts[true_class])
        )
        lines.append(f"- {true_class}: {counts}")

    lines.extend(["", "## Terminal Class Confusion Counts", "", "These counts average the last third of each trajectory.", ""])
    for true_class in sorted(summary.terminal_confusion_counts):
        counts = ", ".join(
            f"{pred}={summary.terminal_confusion_counts[true_class][pred]}"
            for pred in sorted(summary.terminal_confusion_counts[true_class])
        )
        lines.append(f"- {true_class}: {counts}")

    lines.extend(["", "## Scenario Confusion Counts", ""])
    for scenario_name in sorted(summary.scenario_confusion_counts):
        counts = ", ".join(
            f"{pred}={summary.scenario_confusion_counts[scenario_name][pred]}"
            for pred in sorted(summary.scenario_confusion_counts[scenario_name])
        )
        lines.append(f"- {scenario_name}: {counts}")

    lines.extend(["", "## Phase Confusion Counts", ""])
    for phase_name in PHASE_NAMES:
        counts = ", ".join(
            f"{detected}={summary.phase_confusion_counts[phase_name][detected]}"
            for detected in PHASE_NAMES
        )
        lines.append(f"- {phase_name}: {counts}")

    lines.extend(["", "## Scenario Phase Target Hit Rates", "", "These scores compare detected phases against the intended scenario phase windows.", ""])
    for scenario_name in sorted(summary.scenario_phase_total_counts):
        formatted_parts: list[str] = []
        for phase_name in PHASE_NAMES:
            total = summary.scenario_phase_total_counts[scenario_name][phase_name]
            if total == 0:
                continue
            hits = summary.scenario_phase_hit_counts[scenario_name][phase_name]
            formatted_parts.append(f"{phase_name}={hits}/{total} ({hits / total:.2f})")
        lines.append(f"- {scenario_name}: {', '.join(formatted_parts) if formatted_parts else 'no targets'}")

    lines.extend(["", "## Feature Semantics", ""])
    for feature_name in FEATURE_NAMES:
        lines.append(f"- {feature_name}: {FEATURE_DESCRIPTIONS[feature_name]}")

    lines.extend(["", "## Expected Feature-Class Relations", ""])
    for class_name in sorted(summary.per_class_accuracy):
        expected = ", ".join(CLASS_FEATURE_PRIORS.get(class_name, ())) or "no strong prior feature"
        lines.append(f"- {class_name}: expected anchors -> {expected}")

    lines.extend(["", "## Feature-Class Precision and Recall", "", "Precision asks which class a detected feature most often points to. Recall asks how often the class exposes that feature when it should.", ""])
    for class_name in sorted(summary.per_class_accuracy):
        precision_top = _top_feature_list(summary.class_feature_precision[class_name], minimum=0.15)
        recall_top = _top_feature_list(summary.class_feature_recall[class_name], minimum=0.15)
        precision_text = ", ".join(f"{feature}={value:.2f}" for feature, value in precision_top) if precision_top else "none"
        recall_text = ", ".join(f"{feature}={value:.2f}" for feature, value in recall_top) if recall_top else "none"
        lines.append(f"- {class_name}: precision -> {precision_text}; recall -> {recall_text}")

    lines.extend(["", "## Feature Lift by Class", "", "Lift above 1.0 means the feature appears in that class more often than the benchmark average.", ""])
    for feature_name in FEATURE_NAMES:
        top_classes = _top_feature_list(summary.feature_class_lift[feature_name], minimum=0.0)
        top_text = ", ".join(f"{class_name}={value:.2f}" for class_name, value in top_classes) if top_classes else "none"
        lines.append(f"- {feature_name}: {top_text}")

    lines.extend(["", "## True Feature vs Predicted Class Matrix", "", "Rows are true features present in a run. Columns are the final aggregate class assigned to that run.", ""])
    for feature_name in FEATURE_NAMES:
        counts = ", ".join(
            f"{predicted_class}={summary.true_feature_predicted_class_counts[feature_name][predicted_class]}"
            for predicted_class in sorted(summary.true_feature_predicted_class_counts[feature_name])
        )
        lines.append(f"- {feature_name}: {counts}")

    lines.extend(["", "## Detected Feature vs Predicted Class Matrix", "", "Rows are detected features from the classifier. Columns are the final aggregate class assigned to that run.", ""])
    for feature_name in FEATURE_NAMES:
        counts = ", ".join(
            f"{predicted_class}={summary.detected_feature_predicted_class_counts[feature_name][predicted_class]}"
            for predicted_class in sorted(summary.detected_feature_predicted_class_counts[feature_name])
        )
        lines.append(f"- {feature_name}: {counts}")

    lines.extend(["", "## Feature Interpretation Notes", ""])
    for class_name in sorted(summary.per_class_accuracy):
        expected_features = CLASS_FEATURE_PRIORS.get(class_name, ())
        aligned: list[str] = []
        missing: list[str] = []
        for feature_name in expected_features:
            recall_value = summary.class_feature_recall[class_name].get(feature_name, 0.0)
            if recall_value >= 0.40:
                aligned.append(f"{feature_name} ({recall_value:.2f})")
            else:
                missing.append(f"{feature_name} ({recall_value:.2f})")
        aligned_text = ", ".join(aligned) if aligned else "none"
        missing_text = ", ".join(missing) if missing else "none"
        lines.append(f"- {class_name}: aligned -> {aligned_text}; weak or missing -> {missing_text}")

    lines.extend(["", "## Class vs Detected Feature Counts", ""])
    for class_name in sorted(summary.class_feature_detection_counts):
        counts = ", ".join(
            f"{feature}={summary.class_feature_detection_counts[class_name][feature]}"
            for feature in FEATURE_NAMES
        )
        lines.append(f"- {class_name}: {counts}")

    lines.extend(["", "## Feature Confusion Summary", ""])
    for feature_name in FEATURE_NAMES:
        counts = summary.feature_confusion_counts[feature_name]
        lines.append(
            f"- {feature_name}: tp={counts['tp']}, fp={counts['fp']}, fn={counts['fn']}, tn={counts['tn']}"
        )

    lines.extend(["", "## Mean Posterior Entropy by Step", ""])
    for step_index, (mean_entropy, p90_entropy) in enumerate(zip(summary.entropy_mean_by_step, summary.entropy_p90_by_step), start=1):
        lines.append(f"- step {step_index}: mean={mean_entropy:.3f}, p90={p90_entropy:.3f}")

    lines.extend(["", "## Mean Feature Probability by Step", ""])
    for feature_name in FEATURE_NAMES:
        formatted = ", ".join(
            f"s{step_index + 1}={value:.3f}"
            for step_index, value in enumerate(summary.mean_feature_probability_by_step[feature_name])
        )
        lines.append(f"- {feature_name}: {formatted}")

    lines.extend(["", "## Representative Posterior Traces", ""])
    for class_name in sorted(summary.representative_traces):
        lines.append(f"### {class_name}")
        for index, weights in enumerate(summary.representative_traces[class_name], start=1):
            formatted = ", ".join(f"{name}={value:.3f}" for name, value in sorted(weights.items()))
            lines.append(f"- step -{len(summary.representative_traces[class_name]) - index + 1}: {formatted}")
        lines.append("")
    return "\n".join(lines)


def run_toy_benchmark(
    *,
    seed: int = 7,
    steps: int = 32,
    tracks_per_class: int = 8,
    obs_sigma: float = 0.75,
    config: ToyBenchmarkConfig | None = None,
) -> ToyBenchmarkResult:
    benchmark_config = config or ToyBenchmarkConfig(steps=steps, obs_sigma=obs_sigma)
    dataset = generate_toy_dataset(
        config=benchmark_config,
        tracks_per_class=tracks_per_class,
        seed=seed,
    )
    runs = tuple(run_class_bank(track, dataset.class_specs, benchmark_config) for track in dataset.tracks)
    summary = summarize_runs(runs, representative_trace_steps=benchmark_config.representative_trace_steps)
    representative_runs: dict[str, ClassificationRun] = {}
    for run in runs:
        representative_runs.setdefault(run.true_class, run)
    return ToyBenchmarkResult(
        config=benchmark_config,
        dataset=dataset,
        runs=runs,
        summary=summary,
        representative_runs=representative_runs,
    )


def write_toy_benchmark_artifact(
    output_dir: str | Path,
    *,
    seed: int = 7,
    steps: int = 32,
    tracks_per_class: int = 8,
    obs_sigma: float = 0.75,
) -> Path:
    result = run_toy_benchmark(
        seed=seed,
        steps=steps,
        tracks_per_class=tracks_per_class,
        obs_sigma=obs_sigma,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "toy_1d_benchmark_summary.md"
    output_path.write_text(render_toy_benchmark_markdown(result), encoding="utf-8")
    return output_path


def write_toy_benchmark_plot_artifact(
    output_dir: str | Path,
    *,
    seed: int = 7,
    steps: int = 32,
    tracks_per_class: int = 8,
    obs_sigma: float = 0.75,
) -> Path:
    svg_path, _ = write_toy_benchmark_plot_artifacts(
        output_dir,
        seed=seed,
        steps=steps,
        tracks_per_class=tracks_per_class,
        obs_sigma=obs_sigma,
    )
    return svg_path


def write_toy_benchmark_plot_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    steps: int = 32,
    tracks_per_class: int = 8,
    obs_sigma: float = 0.75,
    result: ToyBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_toy_benchmark(
        seed=seed,
        steps=steps,
        tracks_per_class=tracks_per_class,
        obs_sigma=obs_sigma,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    svg_path = output_root / "toy_1d_benchmark_posteriors.svg"
    png_path = output_root / "toy_1d_benchmark_posteriors.png"
    svg_path.write_text(render_toy_benchmark_svg(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_toy_benchmark_png_bytes(benchmark_result))
    return svg_path, png_path


def write_toy_feature_confusion_artifacts(
    output_dir: str | Path,
    *,
    seed: int = 7,
    steps: int = 32,
    tracks_per_class: int = 6,
    obs_sigma: float = 0.75,
    result: ToyBenchmarkResult | None = None,
) -> tuple[Path, Path]:
    benchmark_result = result or run_toy_benchmark(
        seed=seed,
        steps=steps,
        tracks_per_class=tracks_per_class,
        obs_sigma=obs_sigma,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    svg_path = output_root / "toy_1d_feature_confusion.svg"
    png_path = output_root / "toy_1d_feature_confusion.png"
    svg_path.write_text(render_toy_feature_confusion_svg(benchmark_result), encoding="utf-8")
    png_path.write_bytes(render_toy_feature_confusion_png_bytes(benchmark_result))
    return svg_path, png_path


def _format_summary(summary: BenchmarkSummary) -> str:
    lines = [
        "# 1D Toy Bayesian Benchmark",
        "",
        f"Total runs: {summary.total_runs}",
        f"Overall accuracy: {summary.overall_accuracy:.3f}",
        f"Transient accuracy: {summary.transient_accuracy:.3f}",
        f"Terminal accuracy: {summary.terminal_accuracy:.3f}",
        f"Unknown retention mean: {summary.unknown_retention_mean:.3f}",
        "",
        "Per-class accuracy:",
    ]
    for class_name in sorted(summary.per_class_accuracy):
        lines.append(f"- {class_name}: {summary.per_class_accuracy[class_name]:.3f}")

    lines.extend(["", "Confusion counts:"])
    for true_class in sorted(summary.confusion_counts):
        counts = ", ".join(
            f"{pred}={summary.confusion_counts[true_class][pred]}"
            for pred in sorted(summary.confusion_counts[true_class])
        )
        lines.append(f"- {true_class}: {counts}")

    lines.extend(["", "Feature confusion counts:"])
    for feature_name in FEATURE_NAMES:
        counts = summary.feature_confusion_counts[feature_name]
        lines.append(
            f"- {feature_name}: tp={counts['tp']}, fp={counts['fp']}, fn={counts['fn']}, tn={counts['tn']}"
        )

    lines.extend(["", "Mean entropy by step:"])
    for step_index, value in enumerate(summary.entropy_mean_by_step, start=1):
        lines.append(f"- step {step_index}: {value:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 1D toy Bayesian tracking benchmark.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--tracks-per-class", type=int, default=6)
    parser.add_argument("--obs-sigma", type=float, default=0.75)
    args = parser.parse_args(argv)

    result = run_toy_benchmark(
        seed=args.seed,
        steps=args.steps,
        tracks_per_class=args.tracks_per_class,
        obs_sigma=args.obs_sigma,
    )
    print(_format_summary(result.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
