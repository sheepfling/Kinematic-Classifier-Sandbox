from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PlanarKinematics:
    velocity_mps: FloatArray
    acceleration_mps2: FloatArray
    speed_mps: FloatArray
    heading_rad: FloatArray
    yaw_rate_radps: FloatArray
    curvature_per_m: FloatArray
####


def _validate_times_and_vectors(timestamps_s: FloatArray, vectors: FloatArray) -> None:
    if timestamps_s.ndim != 1:
        raise ValueError("timestamps_s must be one-dimensional")
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape N x 3")
    if vectors.shape[0] != timestamps_s.shape[0]:
        raise ValueError("vectors must have one row per timestamp")
    if timestamps_s.shape[0] < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(timestamps_s)) or not np.all(np.isfinite(vectors)):
        raise ValueError("timestamps and vectors must be finite")
    if not np.all(np.diff(timestamps_s) > 0.0):
        raise ValueError("timestamps_s must be strictly increasing")
####


def differentiate_vectors(timestamps_s: FloatArray, vectors: FloatArray) -> FloatArray:
    _validate_times_and_vectors(timestamps_s, vectors)
    derivative = np.gradient(vectors, timestamps_s, axis=0, edge_order=1)
    return np.asarray(derivative, dtype=np.float64)
####


def _stable_heading(velocity_mps: FloatArray, *, minimum_speed_mps: float) -> FloatArray:
    speed_mps = np.linalg.norm(velocity_mps[:, :2], axis=1)
    moving = speed_mps >= minimum_speed_mps
    heading_rad = np.zeros(speed_mps.shape[0], dtype=np.float64)
    if not np.any(moving):
        return heading_rad

    moving_headings = np.unwrap(np.arctan2(velocity_mps[moving, 1], velocity_mps[moving, 0]))
    moving_indices = np.flatnonzero(moving)
    heading_rad[moving_indices] = moving_headings

    first_moving_index = int(moving_indices[0])
    heading_rad[:first_moving_index] = heading_rad[first_moving_index]
    for index in range(first_moving_index + 1, heading_rad.shape[0]):
        if not moving[index]:
            heading_rad[index] = heading_rad[index - 1]
    return heading_rad
####


def derive_planar_kinematics(
    timestamps_s: FloatArray,
    position_m: FloatArray,
    *,
    minimum_speed_mps: float = 0.05,
) -> PlanarKinematics:
    if minimum_speed_mps <= 0.0:
        raise ValueError("minimum_speed_mps must be positive")
    _validate_times_and_vectors(timestamps_s, position_m)

    velocity_mps = differentiate_vectors(timestamps_s, position_m)
    acceleration_mps2 = differentiate_vectors(timestamps_s, velocity_mps)
    speed_mps = np.linalg.norm(velocity_mps[:, :2], axis=1)
    heading_rad = _stable_heading(velocity_mps, minimum_speed_mps=minimum_speed_mps)
    yaw_rate_radps = np.gradient(heading_rad, timestamps_s, edge_order=1)
    curvature_per_m = np.divide(
        yaw_rate_radps,
        speed_mps,
        out=np.zeros_like(yaw_rate_radps, dtype=np.float64),
        where=speed_mps >= minimum_speed_mps,
    )
    return PlanarKinematics(
        velocity_mps=np.asarray(velocity_mps, dtype=np.float64),
        acceleration_mps2=np.asarray(acceleration_mps2, dtype=np.float64),
        speed_mps=np.asarray(speed_mps, dtype=np.float64),
        heading_rad=np.asarray(heading_rad, dtype=np.float64),
        yaw_rate_radps=np.asarray(yaw_rate_radps, dtype=np.float64),
        curvature_per_m=np.asarray(curvature_per_m, dtype=np.float64),
    )
####


__all__ = ["PlanarKinematics", "derive_planar_kinematics", "differentiate_vectors"]
