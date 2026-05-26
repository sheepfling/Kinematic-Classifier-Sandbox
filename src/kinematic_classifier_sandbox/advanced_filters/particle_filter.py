from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy.random as random
from numpy import average, exp, float64, full, log, maximum
from numpy import sum as nsum

from ..utils.types import FloatArray, LogLikelihoodFn, TransitionFn
from .resampling import effective_sample_size, normalize_log_weights, systematic_resample


@dataclass(frozen=True, slots=True)
class ParticleFilterConfig:
    particle_count: int = 512
    resample_threshold_fraction: float = 0.5
    seed: int = 0


@dataclass(slots=True)
class ParticleFilterState:
    trajectory_id: str
    particles: FloatArray
    log_weights: FloatArray
    last_time: float | None = None


@dataclass(frozen=True, slots=True)
class ParticleFilterStep:
    trajectory_id: str
    time: float
    log_marginal_likelihood: float
    state_mean: FloatArray
    state_covariance: FloatArray
    ess: float
    resampled: bool
    weight_entropy: float


class ParticleFilterWeightUpdate(NamedTuple):
    weights: FloatArray
    normalized_log_weights: FloatArray
    log_marginal_likelihood: float
    log_likelihood: FloatArray


def particle_filter_importance_weight_update(
    prior_log_weights: FloatArray,
    propagated_particles: FloatArray,
    observation: FloatArray,
    log_likelihood_fn: LogLikelihoodFn,
) -> ParticleFilterWeightUpdate:
    log_likelihood = log_likelihood_fn(propagated_particles, observation)
    log_unnormalized = prior_log_weights + log_likelihood
    normalization = normalize_log_weights(log_unnormalized)
    return ParticleFilterWeightUpdate(
        weights=normalization.weights,
        normalized_log_weights=normalization.normalized_log_weights,
        log_marginal_likelihood=float(normalization.log_norm),
        log_likelihood=log_likelihood,
    )


class BootstrapParticleFilter:
    def __init__(
        self,
        config: ParticleFilterConfig,
        transition_fn: TransitionFn,
        log_likelihood_fn: LogLikelihoodFn,
    ) -> None:
        self.config = config
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.rng = random.default_rng(config.seed)
        self.state: ParticleFilterState | None = None

    def reset(self, trajectory_id: str, initial_particles: FloatArray) -> None:
        n_particles = initial_particles.shape[0]
        if n_particles != self.config.particle_count:
            raise ValueError("initial_particles must match particle_count")
        self.state = ParticleFilterState(
            trajectory_id=trajectory_id,
            particles=initial_particles.astype(float64),
            log_weights=full(n_particles, -log(n_particles), dtype=float64),
            last_time=None,
        )

    def update(self, time: float, observation: FloatArray) -> ParticleFilterStep:
        if self.state is None:
            raise RuntimeError("Particle filter must be reset before update")
        dt = 0.0 if self.state.last_time is None else max(float(time - self.state.last_time), 1.0e-9)
        propagated = self.transition_fn(self.state.particles, dt, self.rng)
        weight_update = particle_filter_importance_weight_update(
            self.state.log_weights,
            propagated,
            observation,
            self.log_likelihood_fn,
        )
        weights = weight_update.weights
        normalized_log_weights = weight_update.normalized_log_weights
        log_marginal_likelihood = weight_update.log_marginal_likelihood
        ess = effective_sample_size(weights)
        resampled = False
        if ess < self.config.resample_threshold_fraction * self.config.particle_count:
            indexes = systematic_resample(weights, self.rng)
            propagated = propagated[indexes]
            normalized_log_weights = full(
                self.config.particle_count,
                -log(self.config.particle_count),
                dtype=float64,
            )
            weights = exp(normalized_log_weights)
            ess = effective_sample_size(weights)
            resampled = True
        self.state.particles = propagated
        self.state.log_weights = normalized_log_weights
        self.state.last_time = float(time)
        state_mean = average(propagated, axis=0, weights=weights)
        centered = propagated - state_mean
        state_covariance = (centered.T * weights) @ centered
        weight_entropy = -float(nsum(weights * log(maximum(weights, 1.0e-300))))
        return ParticleFilterStep(
            trajectory_id=self.state.trajectory_id,
            time=float(time),
            log_marginal_likelihood=float(log_marginal_likelihood),
            state_mean=state_mean,
            state_covariance=state_covariance,
            ess=float(ess),
            resampled=bool(resampled),
            weight_entropy=weight_entropy,
        )
