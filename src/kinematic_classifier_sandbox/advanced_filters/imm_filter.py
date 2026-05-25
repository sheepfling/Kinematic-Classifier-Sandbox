from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .contracts import AdvancedFilterStep, AdvancedStateSummary
from .linear_gaussian import (
    KalmanModeState,
    LinearGaussianModeSpec,
    kalman_predict,
    kalman_update,
)
from .resampling import logsumexp


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class IMMState:
    mode_probabilities: FloatArray
    mode_states: dict[str, KalmanModeState]
    last_time: float | None
    trajectory_id: str
    latest_mixing_probabilities: dict[str, FloatArray]


class IMMFilter:
    def __init__(
        self,
        mode_specs: list[LinearGaussianModeSpec],
        transition_matrix: FloatArray,
        initial_mode_probabilities: FloatArray | None = None,
        *,
        filter_id: str = "imm_v1",
    ) -> None:
        self.filter_id = filter_id
        self.mode_specs = {spec.mode_id: spec for spec in mode_specs}
        self.mode_ids = [spec.mode_id for spec in mode_specs]
        self.transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        if self.transition_matrix.shape != (len(self.mode_ids), len(self.mode_ids)):
            raise ValueError("transition_matrix shape must match mode_specs")
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("transition_matrix rows must sum to one")
        if initial_mode_probabilities is None:
            n_modes = len(self.mode_ids)
            initial_mode_probabilities = np.full(n_modes, 1.0 / n_modes, dtype=np.float64)
        self.initial_mode_probabilities = np.asarray(initial_mode_probabilities, dtype=np.float64)
        self.initial_mode_probabilities = self.initial_mode_probabilities / np.sum(self.initial_mode_probabilities)
        self.state: IMMState | None = None

    def reset(self, trajectory_id: str, initial_observation: FloatArray | None = None) -> None:
        initial_position = 0.0
        if initial_observation is not None:
            initial_position = float(initial_observation[0])
        mode_states = {
            mode_id: KalmanModeState(
                mean=np.array([initial_position, 0.0, 0.0], dtype=np.float64),
                covariance=np.diag([2.0, 4.0, 4.0]).astype(np.float64),
            )
            for mode_id in self.mode_ids
        }
        self.state = IMMState(
            mode_probabilities=self.initial_mode_probabilities.copy(),
            mode_states=mode_states,
            last_time=None,
            trajectory_id=trajectory_id,
            latest_mixing_probabilities={},
        )

    def update(self, time: float, observation: FloatArray) -> AdvancedFilterStep:
        if self.state is None:
            self.reset(trajectory_id="unknown", initial_observation=observation)
        assert self.state is not None
        dt = 1.0 if self.state.last_time is None else max(float(time - self.state.last_time), 1.0e-9)
        mixed_probabilities = self.transition_matrix.T @ self.state.mode_probabilities
        mixed_probabilities = np.maximum(mixed_probabilities, 1.0e-300)

        mixed_states: dict[str, KalmanModeState] = {}
        mixing_prob_by_dest: dict[str, FloatArray] = {}
        source_means = np.stack([self.state.mode_states[mode_id].mean for mode_id in self.mode_ids])
        for dest_index, dest_mode in enumerate(self.mode_ids):
            numerator = self.state.mode_probabilities * self.transition_matrix[:, dest_index]
            omega = numerator / mixed_probabilities[dest_index]
            mixing_prob_by_dest[dest_mode] = omega
            mixed_mean = np.sum(omega[:, None] * source_means, axis=0)
            mixed_cov = np.zeros_like(self.state.mode_states[dest_mode].covariance)
            for src_index, src_mode in enumerate(self.mode_ids):
                src_state = self.state.mode_states[src_mode]
                delta = (src_state.mean - mixed_mean).reshape(-1, 1)
                mixed_cov += omega[src_index] * (src_state.covariance + delta @ delta.T)
            mixed_states[dest_mode] = KalmanModeState(mean=mixed_mean, covariance=mixed_cov)

        log_likelihoods: list[float] = []
        updated_states: dict[str, KalmanModeState] = {}
        innovations: dict[str, float] = {}
        for mode_id in self.mode_ids:
            spec = self.mode_specs[mode_id]
            predicted = kalman_predict(
                mixed_states[mode_id],
                dt=dt,
                process_noise_scale=spec.process_noise_scale,
                acceleration_bias=spec.acceleration_bias,
            )
            updated, log_likelihood, innovation, _ = kalman_update(
                predicted,
                observation=observation,
                measurement_noise=spec.measurement_noise,
            )
            updated_states[mode_id] = updated
            log_likelihoods.append(float(log_likelihood))
            innovations[mode_id] = float(innovation[0])

        log_prior = np.log(mixed_probabilities)
        log_numerators = log_prior + np.asarray(log_likelihoods, dtype=np.float64)
        log_norm = logsumexp(log_numerators)
        mode_probabilities = np.exp(log_numerators - log_norm)
        self.state.mode_probabilities = mode_probabilities
        self.state.mode_states = updated_states
        self.state.last_time = float(time)
        self.state.latest_mixing_probabilities = mixing_prob_by_dest
        best_index = int(np.argmax(mode_probabilities))
        predicted_mode = self.mode_ids[best_index]
        entropy = -float(np.sum(mode_probabilities * np.log(np.maximum(mode_probabilities, 1.0e-300))))
        return AdvancedFilterStep(
            trajectory_id=self.state.trajectory_id,
            time=float(time),
            filter_id=self.filter_id,
            predicted_label=predicted_mode,
            confidence=float(mode_probabilities[best_index]),
            posterior_by_label={
                mode_id: float(mode_probabilities[index])
                for index, mode_id in enumerate(self.mode_ids)
            },
            log_evidence_by_label={
                mode_id: float(log_likelihoods[index])
                for index, mode_id in enumerate(self.mode_ids)
            },
            diagnostics={
                "dt": dt,
                "log_norm": float(log_norm),
                "mode_entropy": entropy,
                "mixing_prob_sum_error_max": float(
                    max(abs(float(np.sum(values)) - 1.0) for values in mixing_prob_by_dest.values())
                ),
                **{f"innovation_{mode_id}": innovations[mode_id] for mode_id in self.mode_ids},
            },
        )

    def state_summary(self) -> AdvancedStateSummary:
        if self.state is None:
            raise RuntimeError("IMMFilter must be reset before state_summary")
        means = np.stack([self.state.mode_states[mode_id].mean for mode_id in self.mode_ids])
        probs = self.state.mode_probabilities
        combined_mean = np.sum(probs[:, None] * means, axis=0)
        combined_cov = np.zeros_like(self.state.mode_states[self.mode_ids[0]].covariance)
        for index, mode_id in enumerate(self.mode_ids):
            mode_state = self.state.mode_states[mode_id]
            delta = (mode_state.mean - combined_mean).reshape(-1, 1)
            combined_cov += probs[index] * (mode_state.covariance + delta @ delta.T)
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1.0e-300))))
        return AdvancedStateSummary(
            trajectory_id=self.state.trajectory_id,
            time=-1.0 if self.state.last_time is None else self.state.last_time,
            filter_id=self.filter_id,
            state_mean=combined_mean,
            state_covariance=combined_cov,
            diagnostics={"mode_entropy": entropy},
        )
