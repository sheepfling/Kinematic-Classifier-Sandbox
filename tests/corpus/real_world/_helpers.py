from __future__ import annotations

import numpy as np

from kinematic_classifier_sandbox.corpus.real_world.contracts import (
    LabelEvidence,
    NormalizedTrack,
    TrackLabels,
    TrackProvenance,
)


def make_real_world_track(
    *,
    split_group_id: str,
    normalized_class: str,
    duration_s: float = 40.0,
    sample_interval_s: float = 0.1,
    speed_mps: float = 2.0,
    vertical_speed_mps: float = 0.0,
    speed_axis_count: int = 2,
    gap_after_s: float | None = None,
    gap_duration_s: float = 1.0,
) -> NormalizedTrack:
    sample_count = int(round(duration_s / sample_interval_s)) + 1
    timestamps_s = np.arange(sample_count, dtype=np.float64) * sample_interval_s
    if gap_after_s is not None:
        gap_index = int(round(gap_after_s / sample_interval_s)) + 1
        timestamps_s[gap_index:] += gap_duration_s

    x_m = speed_mps * timestamps_s
    z_m = vertical_speed_mps * timestamps_s
    position_m = np.column_stack(
        (
            x_m,
            np.zeros(sample_count, dtype=np.float64),
            z_m,
        )
    )
    velocity_mps = np.column_stack(
        (
            np.full(sample_count, speed_mps, dtype=np.float64),
            np.zeros(sample_count, dtype=np.float64),
            np.full(sample_count, vertical_speed_mps, dtype=np.float64),
        )
    )
    acceleration_mps2 = np.zeros_like(velocity_mps)
    native_label = "3" if normalized_class == "passenger_car" else "7"

    return NormalizedTrack(
        provenance=TrackProvenance(
            dataset_id="tgsim_foggy_bottom",
            source_asset_id="trajectory_csv",
            recording_id="foggy_bottom_2023-05-04",
            run_id="run_1",
            track_id=split_group_id,
            location_id="foggy_bottom_washington_dc",
            split_group_id=split_group_id,
        ),
        labels=TrackLabels(
            native_label=native_label,
            normalized_class=normalized_class,
            mobility_family="conventional_steering",
            operating_domain="urban_road",
            evidence=LabelEvidence.NATIVE,
        ),
        coordinate_frame="foggy_bottom_local_metric_xy",
        speed_axis_count=speed_axis_count,
        timestamps_s=timestamps_s,
        position_m=position_m,
        derived_velocity_mps=velocity_mps,
        derived_acceleration_mps2=acceleration_mps2,
        numeric_channels=(),
        categorical_channels=(),
        quality=None,
        metadata={},
    )
####


__all__ = ["make_real_world_track"]
