from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .contracts import NormalizedTrack, TrackQuality


class TrackQualityPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nominal_sample_interval_s: float = Field(default=0.1, gt=0.0)
    gap_multiplier: float = Field(default=2.5, gt=1.0)
    max_position_step_m: float = Field(default=20.0, gt=0.0)
    source_velocity_rmse_warning_mps: float = Field(default=5.0, gt=0.0)
    source_acceleration_rmse_warning_mps2: float = Field(default=10.0, gt=0.0)
####


def _vector_rmse(left: np.ndarray, right: np.ndarray) -> float:
    residual = left - right
    squared_norm = np.sum(np.square(residual), axis=1)
    return float(np.sqrt(np.mean(squared_norm)))
####


def assess_track_quality(
    track: NormalizedTrack,
    *,
    policy: TrackQualityPolicy | None = None,
) -> TrackQuality:
    effective_policy = policy or TrackQualityPolicy()
    time_steps = np.diff(track.timestamps_s)
    position_steps = np.linalg.norm(np.diff(track.position_m, axis=0), axis=1)

    source_velocity_rmse_mps: float | None = None
    if track.source_velocity_mps is not None:
        source_velocity_rmse_mps = _vector_rmse(
            track.source_velocity_mps,
            track.derived_velocity_mps,
        )

    source_acceleration_rmse_mps2: float | None = None
    if track.source_acceleration_mps2 is not None:
        source_acceleration_rmse_mps2 = _vector_rmse(
            track.source_acceleration_mps2,
            track.derived_acceleration_mps2,
        )

    gap_threshold_s = (
        effective_policy.nominal_sample_interval_s * effective_policy.gap_multiplier
    )
    gap_count = int(np.count_nonzero(time_steps > gap_threshold_s))
    maximum_position_step_m = float(np.max(position_steps))

    findings: list[str] = []
    if gap_count:
        findings.append(
            f"{gap_count} sample gap(s) exceeded {gap_threshold_s:.6g} seconds"
        )
    if maximum_position_step_m > effective_policy.max_position_step_m:
        findings.append(
            "maximum position step exceeded "
            f"{effective_policy.max_position_step_m:.6g} meters"
        )
    if (
        source_velocity_rmse_mps is not None
        and source_velocity_rmse_mps
        > effective_policy.source_velocity_rmse_warning_mps
    ):
        findings.append(
            "source-versus-derived velocity RMSE exceeded "
            f"{effective_policy.source_velocity_rmse_warning_mps:.6g} m/s"
        )
    if (
        source_acceleration_rmse_mps2 is not None
        and source_acceleration_rmse_mps2
        > effective_policy.source_acceleration_rmse_warning_mps2
    ):
        findings.append(
            "source-versus-derived acceleration RMSE exceeded "
            f"{effective_policy.source_acceleration_rmse_warning_mps2:.6g} m/s^2"
        )

    return TrackQuality(
        sample_count=int(track.timestamps_s.shape[0]),
        duration_s=float(track.timestamps_s[-1] - track.timestamps_s[0]),
        median_dt_s=float(np.median(time_steps)),
        max_dt_s=float(np.max(time_steps)),
        gap_count=gap_count,
        max_position_step_m=maximum_position_step_m,
        source_velocity_rmse_mps=source_velocity_rmse_mps,
        source_acceleration_rmse_mps2=source_acceleration_rmse_mps2,
        findings=tuple(findings),
    )
####


__all__ = ["TrackQualityPolicy", "assess_track_quality"]
