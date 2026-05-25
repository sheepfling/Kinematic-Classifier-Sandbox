from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def make_initial_particles_1d(
    particle_count: int,
    position_mean: float,
    position_std: float,
    velocity_mean: float,
    velocity_std: float,
    rng: np.random.Generator,
) -> FloatArray:
    positions = rng.normal(position_mean, position_std, size=particle_count)
    velocities = rng.normal(velocity_mean, velocity_std, size=particle_count)
    return np.column_stack([positions, velocities]).astype(np.float64)


def nonlinear_drag_transition(
    particles: FloatArray,
    dt: float,
    rng: np.random.Generator,
    acceleration_command: float = 0.0,
    drag_coefficient: float = 0.05,
    process_std: float = 0.05,
) -> FloatArray:
    next_particles = particles.copy()
    position = particles[:, 0]
    velocity = particles[:, 1]
    drag = drag_coefficient * velocity * np.abs(velocity)
    velocity_noise = rng.normal(0.0, process_std, size=len(particles))
    next_velocity = velocity + (acceleration_command - drag) * dt + velocity_noise
    next_position = position + next_velocity * dt
    next_particles[:, 0] = next_position
    next_particles[:, 1] = next_velocity
    return next_particles


def constant_velocity_transition(
    particles: FloatArray,
    dt: float,
    rng: np.random.Generator,
    process_std: float = 0.03,
) -> FloatArray:
    next_particles = particles.copy()
    noise = rng.normal(0.0, process_std, size=particles.shape)
    next_particles[:, 0] = particles[:, 0] + particles[:, 1] * dt + noise[:, 0]
    next_particles[:, 1] = particles[:, 1] + noise[:, 1]
    return next_particles


def position_gaussian_log_likelihood(
    particles: FloatArray,
    observation: FloatArray,
    measurement_std: float,
) -> FloatArray:
    residual = float(observation[0]) - particles[:, 0]
    variance = measurement_std**2
    return -0.5 * (np.log(2.0 * np.pi * variance) + residual**2 / variance)


def position_mixture_log_likelihood(
    particles: FloatArray,
    observation: FloatArray,
    measurement_std: float,
    outlier_std: float,
    outlier_probability: float,
) -> FloatArray:
    residual = float(observation[0]) - particles[:, 0]
    base_var = measurement_std**2
    outlier_var = outlier_std**2
    base_log = np.log(1.0 - outlier_probability) - 0.5 * (
        np.log(2.0 * np.pi * base_var) + residual**2 / base_var
    )
    outlier_log = np.log(outlier_probability) - 0.5 * (
        np.log(2.0 * np.pi * outlier_var) + residual**2 / outlier_var
    )
    max_log = np.maximum(base_log, outlier_log)
    return max_log + np.log(np.exp(base_log - max_log) + np.exp(outlier_log - max_log))
