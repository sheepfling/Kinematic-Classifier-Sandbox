from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearGaussianModeSpec:
    mode_id: str
    process_noise_scale: float
    measurement_noise: float
    acceleration_bias: float = 0.0
    state_dim: int = 3


@dataclass(slots=True)
class KalmanModeState:
    mean: FloatArray
    covariance: FloatArray


def constant_acceleration_transition(dt: float) -> FloatArray:
    return np.array(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def position_measurement_matrix() -> FloatArray:
    return np.array([[1.0, 0.0, 0.0]], dtype=np.float64)


def process_noise(dt: float, scale: float) -> FloatArray:
    q = float(scale)
    return q * np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ],
        dtype=np.float64,
    )


def kalman_predict(
    state: KalmanModeState,
    dt: float,
    process_noise_scale: float,
    *,
    acceleration_bias: float = 0.0,
) -> KalmanModeState:
    f = constant_acceleration_transition(dt)
    q = process_noise(dt, process_noise_scale)
    bias = np.array([0.5 * acceleration_bias * dt * dt, acceleration_bias * dt, acceleration_bias], dtype=np.float64)
    mean = f @ state.mean + bias
    covariance = f @ state.covariance @ f.T + q
    return KalmanModeState(mean=mean, covariance=covariance)


def kalman_update(
    state: KalmanModeState,
    observation: FloatArray,
    measurement_noise: float,
) -> tuple[KalmanModeState, float, FloatArray, FloatArray]:
    h = position_measurement_matrix()
    r = np.array([[measurement_noise]], dtype=np.float64)
    innovation = observation.reshape(1, 1) - h @ state.mean.reshape(-1, 1)
    innovation_covariance = h @ state.covariance @ h.T + r
    innovation_covariance = innovation_covariance + 1.0e-9 * np.eye(innovation_covariance.shape[0])
    s_inv = np.linalg.inv(innovation_covariance)
    gain = state.covariance @ h.T @ s_inv
    mean = state.mean.reshape(-1, 1) + gain @ innovation
    covariance = (np.eye(state.covariance.shape[0]) - gain @ h) @ state.covariance
    log_likelihood = -0.5 * float(
        np.log(2.0 * np.pi * innovation_covariance[0, 0])
        + (innovation.T @ s_inv @ innovation)[0, 0]
    )
    return (
        KalmanModeState(mean=mean.ravel(), covariance=covariance),
        log_likelihood,
        innovation.ravel(),
        innovation_covariance,
    )
