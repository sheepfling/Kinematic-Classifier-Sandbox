from __future__ import annotations

from dataclasses import dataclass

import numpy.random as random
from numpy import array, column_stack, diag, float64, repeat, zeros

from ..validation.shared_evaluation import SharedClassifierRun
from .models_1d import (
    constant_velocity_transition,
    make_initial_particles_1d,
    position_gaussian_log_likelihood,
)
from .particle_filter import BootstrapParticleFilter, ParticleFilterConfig
from .particle_filter_bank import ParticleFilterBank
from .rbpf import LinearModeModel, RBPFConfig, RaoBlackwellizedParticleFilter

CLASS_NAMES = ("constant_velocity", "constant_acceleration")
_ACCELERATION_MEAN = 0.28


@dataclass(frozen=True, slots=True)
class SharedAdvancedRun:
    method_name: str
    final_predicted_class: str
    final_confidence: float
    final_weights: dict[str, float]


def _normalized_prior(prior: dict[str, float] | None) -> dict[str, float]:
    if prior is None:
        return {class_name: 1.0 / len(CLASS_NAMES) for class_name in CLASS_NAMES}
    total = sum(max(float(prior.get(class_name, 0.0)), 0.0) for class_name in CLASS_NAMES)
    if total <= 1.0e-12:
        return {class_name: 1.0 / len(CLASS_NAMES) for class_name in CLASS_NAMES}
    return {
        class_name: max(float(prior.get(class_name, 0.0)), 0.0) / total
        for class_name in CLASS_NAMES
    }


def _initial_velocity_estimate(trajectory: object) -> float:
    times = tuple(float(value) for value in getattr(trajectory, "times"))
    measurements = tuple(float(value) for value in getattr(trajectory, "measurements"))
    if len(times) < 2:
        return 0.8
    dt = max(times[1] - times[0], 1.0e-6)
    return (measurements[1] - measurements[0]) / dt


def _constant_acceleration_transition(
    particles,
    dt: float,
    rng: random.Generator,
    process_std: float = 0.04,
):
    next_particles = particles.copy()
    noise = rng.normal(0.0, process_std, size=particles.shape)
    next_particles[:, 0] = particles[:, 0] + particles[:, 1] * dt + 0.5 * particles[:, 2] * dt * dt + noise[:, 0]
    next_particles[:, 1] = particles[:, 1] + particles[:, 2] * dt + noise[:, 1]
    next_particles[:, 2] = particles[:, 2] + noise[:, 2]
    return next_particles


def _initial_constant_acceleration_particles(
    particle_count: int,
    position_mean: float,
    velocity_mean: float,
    rng: random.Generator,
):
    positions = rng.normal(position_mean, 0.20, size=particle_count)
    velocities = rng.normal(velocity_mean, 0.35, size=particle_count)
    accelerations = rng.normal(_ACCELERATION_MEAN, 0.12, size=particle_count)
    return column_stack([positions, velocities, accelerations]).astype(float64)


def run_shared_particle_filter_classifier(
    trajectory: object,
    *,
    prior: dict[str, float] | None = None,
) -> SharedAdvancedRun:
    prior_by_label = _normalized_prior(prior)
    particle_count = 192
    rng = random.default_rng(int(getattr(trajectory, "seed")) + 1701)
    observation_sigma = 0.20
    filters = {
        "constant_velocity": BootstrapParticleFilter(
            ParticleFilterConfig(particle_count=particle_count, seed=int(getattr(trajectory, "seed")) + 1702),
            transition_fn=lambda particles, dt, gen: constant_velocity_transition(particles, dt, gen, process_std=0.04),
            log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, measurement_std=observation_sigma),
        ),
        "constant_acceleration": BootstrapParticleFilter(
            ParticleFilterConfig(particle_count=particle_count, seed=int(getattr(trajectory, "seed")) + 1703),
            transition_fn=lambda particles, dt, gen: _constant_acceleration_transition(particles, dt, gen, process_std=0.04),
            log_likelihood_fn=lambda particles, obs: position_gaussian_log_likelihood(particles, obs, measurement_std=observation_sigma),
        ),
    }
    bank = ParticleFilterBank(filters, prior_by_label=prior_by_label, filter_id="particle_filter_bank")
    initial_velocity = _initial_velocity_estimate(trajectory)
    first_measurement = float(getattr(trajectory, "measurements")[0])
    initial_particles = {
        "constant_velocity": make_initial_particles_1d(
            particle_count,
            first_measurement,
            0.20,
            initial_velocity,
            0.35,
            rng,
        ),
        "constant_acceleration": _initial_constant_acceleration_particles(
            particle_count,
            first_measurement,
            initial_velocity,
            rng,
        ),
    }
    bank.reset(str(getattr(trajectory, "trajectory_id")), initial_particles)
    final_step = None
    for time_value, measurement in zip(getattr(trajectory, "times"), getattr(trajectory, "measurements"), strict=True):
        final_step = bank.update(float(time_value), array([float(measurement)], dtype=float64))
    assert final_step is not None
    return SharedAdvancedRun(
        method_name="particle_filter_bank",
        final_predicted_class=final_step.predicted_label,
        final_confidence=float(final_step.confidence),
        final_weights=final_step.posterior_by_label,
    )


def _rbpf_mode_models() -> list[LinearModeModel]:
    cv_transition = array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.85],
        ],
        dtype=float64,
    )
    ca_transition = array(
        [
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.93],
        ],
        dtype=float64,
    )
    process_covariance = diag([0.06, 0.05, 0.04]).astype(float64)
    measurement_matrix = array([[1.0, 0.0, 0.0]], dtype=float64)
    measurement_covariance = array([[0.20**2]], dtype=float64)
    return [
        LinearModeModel(
            mode_id="constant_velocity",
            transition_matrix=cv_transition,
            process_covariance=process_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
            control_bias=zeros(3, dtype=float64),
        ),
        LinearModeModel(
            mode_id="constant_acceleration",
            transition_matrix=ca_transition,
            process_covariance=process_covariance,
            measurement_matrix=measurement_matrix,
            measurement_covariance=measurement_covariance,
            control_bias=array([0.0, 0.0, _ACCELERATION_MEAN * 0.07], dtype=float64),
        ),
    ]


def run_shared_rbpf_classifier(
    trajectory: object,
    *,
    prior: dict[str, float] | None = None,
) -> SharedAdvancedRun:
    prior_by_label = _normalized_prior(prior)
    particle_count = 160
    rng = random.default_rng(int(getattr(trajectory, "seed")) + 1801)
    mode_models = _rbpf_mode_models()
    rbpf = RaoBlackwellizedParticleFilter(
        RBPFConfig(particle_count=particle_count, seed=int(getattr(trajectory, "seed")) + 1802),
        mode_models,
        array([[0.95, 0.05], [0.06, 0.94]], dtype=float64),
    )
    first_measurement = float(getattr(trajectory, "measurements")[0])
    initial_velocity = _initial_velocity_estimate(trajectory)
    initial_modes = rng.choice(
        2,
        size=particle_count,
        p=array(
            [
                prior_by_label["constant_velocity"],
                prior_by_label["constant_acceleration"],
            ],
            dtype=float64,
        ),
    )
    means = zeros((particle_count, 3), dtype=float64)
    means[:, 0] = first_measurement
    means[:, 1] = rng.normal(initial_velocity, 0.30, size=particle_count)
    means[:, 2] = rng.normal(0.0, 0.10, size=particle_count)
    means[initial_modes == 1, 2] = rng.normal(_ACCELERATION_MEAN, 0.08, size=int((initial_modes == 1).sum()))
    covariances = repeat(diag([0.20, 0.30, 0.15])[None, :, :], particle_count, axis=0)
    rbpf.reset(str(getattr(trajectory, "trajectory_id")), initial_modes, means, covariances)
    final_step = None
    for time_value, measurement in zip(getattr(trajectory, "times"), getattr(trajectory, "measurements"), strict=True):
        final_step = rbpf.update(float(time_value), array([float(measurement)], dtype=float64))
    assert final_step is not None
    return SharedAdvancedRun(
        method_name="rbpf",
        final_predicted_class=final_step.predicted_label,
        final_confidence=float(final_step.confidence),
        final_weights=final_step.posterior_by_label,
    )


def as_shared_classifier_run(trajectory: object, run: SharedAdvancedRun) -> SharedClassifierRun:
    return SharedClassifierRun(
        method_name=run.method_name,
        sensor_regime_id="position_only",
        trajectory_id=str(getattr(trajectory, "trajectory_id")),
        true_class=str(getattr(trajectory, "true_class")),
        scenario_name=str(getattr(trajectory, "scenario_name")),
        final_predicted_class=run.final_predicted_class,
        final_confidence=run.final_confidence,
        final_weights=run.final_weights,
        measurement_dim=int(getattr(trajectory, "measurement_dim", 1)),
        coordinate_frame=str(getattr(trajectory, "coordinate_frame", "scalar_line")),
    )
