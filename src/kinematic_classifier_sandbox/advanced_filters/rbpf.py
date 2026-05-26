from __future__ import annotations

from dataclasses import dataclass

from numpy import allclose, asarray, exp, eye, float64, full, int64, log, ndarray, pi, sum as nsum, zeros, zeros_like
import numpy.linalg as linalg
import numpy.random as random

from .contracts import AdvancedFilterStep
from .resampling import effective_sample_size, logsumexp, normalize_log_weights, systematic_resample
from ..utils.types import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class LinearModeModel:
    mode_id: str
    transition_matrix: FloatArray
    process_covariance: FloatArray
    measurement_matrix: FloatArray
    measurement_covariance: FloatArray
    control_bias: FloatArray


@dataclass(slots=True)
class RBPFState:
    trajectory_id: str
    mode_indexes: IntArray
    means: FloatArray
    covariances: FloatArray
    log_weights: FloatArray
    last_time: float | None = None


@dataclass(frozen=True, slots=True)
class RBPFConfig:
    particle_count: int = 256
    resample_threshold_fraction: float = 0.5
    seed: int = 0


def rbpf_conditional_weight_update(
    prior_log_weights: FloatArray,
    conditional_log_likelihoods: FloatArray,
) -> tuple[FloatArray, FloatArray, float, FloatArray]:
    log_unnormalized = prior_log_weights + conditional_log_likelihoods
    weights, normalized_log_weights, log_marginal = normalize_log_weights(log_unnormalized)
    return weights, normalized_log_weights, float(log_marginal), log_unnormalized


class RaoBlackwellizedParticleFilter:
    def __init__(
        self,
        config: RBPFConfig,
        mode_models: list[LinearModeModel],
        mode_transition_matrix: FloatArray,
    ) -> None:
        self.filter_id = "rbpf_v1"
        self.config = config
        self.mode_models = mode_models
        self.mode_ids = [model.mode_id for model in mode_models]
        self.mode_transition_matrix = asarray(mode_transition_matrix, dtype=float64)
        if not allclose(self.mode_transition_matrix.sum(axis=1), 1.0):
            raise ValueError("mode_transition_matrix rows must sum to one")
        self.rng = random.default_rng(config.seed)
        self.state: RBPFState | None = None

    def reset(
        self,
        trajectory_id: str,
        initial_mode_indexes: IntArray,
        initial_means: FloatArray,
        initial_covariances: FloatArray,
    ) -> None:
        n_particles = len(initial_mode_indexes)
        if n_particles != self.config.particle_count:
            raise ValueError("initial_mode_indexes must match particle_count")
        self.state = RBPFState(
            trajectory_id=trajectory_id,
            mode_indexes=initial_mode_indexes.astype(int64),
            means=initial_means.astype(float64),
            covariances=initial_covariances.astype(float64),
            log_weights=full(n_particles, -log(n_particles), dtype=float64),
            last_time=None,
        )

    def update(self, time: float, observation: FloatArray) -> AdvancedFilterStep:
        if self.state is None:
            raise RuntimeError("RBPF must be reset before update")
        new_mode_indexes = self._sample_modes(self.state.mode_indexes)
        new_means = zeros_like(self.state.means)
        new_covariances = zeros_like(self.state.covariances)
        log_likelihoods = zeros(self.config.particle_count, dtype=float64)
        for index in range(self.config.particle_count):
            model = self.mode_models[int(new_mode_indexes[index])]
            mean, covariance, log_likelihood = kalman_predict_update(
                mean=self.state.means[index],
                covariance=self.state.covariances[index],
                observation=observation,
                model=model,
            )
            new_means[index] = mean
            new_covariances[index] = covariance
            log_likelihoods[index] = log_likelihood
        weights, normalized_log_weights, log_marginal, log_unnormalized = rbpf_conditional_weight_update(
            self.state.log_weights,
            log_likelihoods,
        )
        log_evidence_by_mode = self._mode_log_evidence(log_unnormalized, new_mode_indexes)
        ess = effective_sample_size(weights)
        resampled = False
        if ess < self.config.resample_threshold_fraction * self.config.particle_count:
            indexes = systematic_resample(weights, self.rng)
            new_mode_indexes = new_mode_indexes[indexes]
            new_means = new_means[indexes]
            new_covariances = new_covariances[indexes]
            normalized_log_weights = full(
                self.config.particle_count,
                -log(self.config.particle_count),
                dtype=float64,
            )
            weights = exp(normalized_log_weights)
            ess = effective_sample_size(weights)
            resampled = True
        self.state.mode_indexes = new_mode_indexes
        self.state.means = new_means
        self.state.covariances = new_covariances
        self.state.log_weights = normalized_log_weights
        self.state.last_time = float(time)
        posterior_by_mode = self._mode_posterior(weights, new_mode_indexes)
        predicted_mode = max(posterior_by_mode, key=posterior_by_mode.get)
        return AdvancedFilterStep(
            trajectory_id=self.state.trajectory_id,
            time=float(time),
            filter_id=self.filter_id,
            predicted_label=predicted_mode,
            confidence=float(posterior_by_mode[predicted_mode]),
            posterior_by_label=posterior_by_mode,
            log_evidence_by_label=log_evidence_by_mode,
            diagnostics={
                "ess": float(ess),
                "resampled": bool(resampled),
                "log_marginal_likelihood": float(log_marginal),
            },
        )

    def _sample_modes(self, old_mode_indexes: IntArray) -> IntArray:
        new_modes = zeros_like(old_mode_indexes)
        for index, old_mode in enumerate(old_mode_indexes):
            probabilities = self.mode_transition_matrix[int(old_mode)]
            new_modes[index] = int(self.rng.choice(len(probabilities), p=probabilities))
        return new_modes

    def _mode_posterior(self, weights: FloatArray, mode_indexes: IntArray) -> dict[str, float]:
        posterior = {mode_id: 0.0 for mode_id in self.mode_ids}
        for mode_index, weight in zip(mode_indexes, weights, strict=True):
            posterior[self.mode_ids[int(mode_index)]] += float(weight)
        return posterior

    def _mode_log_evidence(self, log_unnormalized: FloatArray, mode_indexes: IntArray) -> dict[str, float]:
        evidence: dict[str, float] = {}
        for mode_index, mode_id in enumerate(self.mode_ids):
            values = log_unnormalized[mode_indexes == mode_index]
            evidence[mode_id] = float(logsumexp(values)) if len(values) else float("-inf")
        return evidence


def kalman_predict_update(
    mean: FloatArray,
    covariance: FloatArray,
    observation: FloatArray,
    model: LinearModeModel,
) -> tuple[FloatArray, FloatArray, float]:
    predicted_mean = model.transition_matrix @ mean + model.control_bias
    predicted_covariance = (
        model.transition_matrix @ covariance @ model.transition_matrix.T
        + model.process_covariance
    )
    innovation = observation - model.measurement_matrix @ predicted_mean
    innovation_covariance = (
        model.measurement_matrix @ predicted_covariance @ model.measurement_matrix.T
        + model.measurement_covariance
    )
    innovation_covariance = innovation_covariance + 1.0e-9 * eye(innovation_covariance.shape[0])
    inv_s = linalg.inv(innovation_covariance)
    gain = predicted_covariance @ model.measurement_matrix.T @ inv_s
    updated_mean = predicted_mean + gain @ innovation
    updated_covariance = (eye(predicted_covariance.shape[0]) - gain @ model.measurement_matrix) @ predicted_covariance
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    sign, log_det = linalg.slogdet(innovation_covariance)
    if sign <= 0:
        raise ValueError("innovation covariance must be positive definite")
    mahalanobis = float(innovation.T @ inv_s @ innovation)
    dim = observation.shape[0]
    log_likelihood = -0.5 * (dim * log(2.0 * pi) + log_det + mahalanobis)
    return updated_mean, updated_covariance, float(log_likelihood)
