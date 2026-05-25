from __future__ import annotations

import numpy as np

from .rbpf import LinearModeModel


def transition_ca_1d(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_rbpf_1d_mode_models(dt: float, measurement_std: float) -> list[LinearModeModel]:
    f = transition_ca_1d(dt)
    h = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    r = np.array([[measurement_std**2]], dtype=np.float64)
    return [
        LinearModeModel(
            mode_id="coast",
            transition_matrix=f,
            process_covariance=np.diag([0.005, 0.01, 0.02]).astype(np.float64),
            measurement_matrix=h,
            measurement_covariance=r,
            control_bias=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        LinearModeModel(
            mode_id="accelerate",
            transition_matrix=f,
            process_covariance=np.diag([0.01, 0.02, 0.05]).astype(np.float64),
            measurement_matrix=h,
            measurement_covariance=r,
            control_bias=np.array([0.0, 0.0, 0.22], dtype=np.float64),
        ),
        LinearModeModel(
            mode_id="brake",
            transition_matrix=f,
            process_covariance=np.diag([0.01, 0.02, 0.05]).astype(np.float64),
            measurement_matrix=h,
            measurement_covariance=r,
            control_bias=np.array([0.0, 0.0, -0.22], dtype=np.float64),
        ),
        LinearModeModel(
            mode_id="maneuver",
            transition_matrix=f,
            process_covariance=np.diag([0.03, 0.08, 0.25]).astype(np.float64),
            measurement_matrix=h,
            measurement_covariance=r,
            control_bias=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
    ]


def default_mode_transition_matrix_1d() -> np.ndarray:
    return np.array(
        [
            [0.62, 0.25, 0.03, 0.10],
            [0.06, 0.82, 0.02, 0.10],
            [0.12, 0.04, 0.78, 0.06],
            [0.15, 0.15, 0.10, 0.60],
        ],
        dtype=np.float64,
    )
