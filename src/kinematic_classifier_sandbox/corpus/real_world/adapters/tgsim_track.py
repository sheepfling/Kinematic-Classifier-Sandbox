from __future__ import annotations

import numpy as np

from ..contracts import (
    ChannelRole,
    DatasetManifest,
    NormalizedTrack,
    NumericChannel,
    TrackLabels,
    TrackProvenance,
)
from ..kinematics import derive_planar_kinematics
from ..quality import assess_track_quality
from .tgsim_contracts import TgsimFoggyBottomAdapterConfig, _TgsimRow


def _column_array(rows: list[_TgsimRow], attribute_name: str) -> np.ndarray:
    return np.asarray([float(getattr(row, attribute_name)) for row in rows], dtype=np.float64)
####


def build_tgsim_track(
    rows: list[_TgsimRow],
    *,
    labels: TrackLabels,
    manifest: DatasetManifest,
    config: TgsimFoggyBottomAdapterConfig,
) -> NormalizedTrack:
    timestamps_s = _column_array(rows, "time_s")
    position_m = np.column_stack(
        (
            _column_array(rows, "x_m"),
            _column_array(rows, "y_m"),
            np.zeros(len(rows), dtype=np.float64),
        )
    )
    source_velocity_mps = np.column_stack(
        (
            _column_array(rows, "velocity_x_mps"),
            _column_array(rows, "velocity_y_mps"),
            np.zeros(len(rows), dtype=np.float64),
        )
    )
    source_acceleration_mps2 = np.column_stack(
        (
            _column_array(rows, "acceleration_x_mps2"),
            _column_array(rows, "acceleration_y_mps2"),
            np.zeros(len(rows), dtype=np.float64),
        )
    )
    derived = derive_planar_kinematics(
        timestamps_s,
        position_m,
        minimum_speed_mps=config.minimum_heading_speed_mps,
    )
    source_speed_mps = np.linalg.norm(source_velocity_mps[:, :2], axis=1)

    first_row = min(row.source_row_number for row in rows)
    last_row = max(row.source_row_number for row in rows)
    run_id = rows[0].run_id
    track_id = rows[0].track_id
    split_group_id = ":".join(
        (manifest.dataset_id, config.recording_id, run_id, track_id)
    )

    track = NormalizedTrack(
        provenance=TrackProvenance(
            dataset_id=manifest.dataset_id,
            source_asset_id=config.source_asset_id,
            recording_id=config.recording_id,
            run_id=run_id,
            track_id=track_id,
            location_id=config.location_id,
            split_group_id=split_group_id,
            source_row_start=first_row,
            source_row_end=last_row,
        ),
        labels=labels,
        coordinate_frame=manifest.coordinate_frame,
        timestamps_s=timestamps_s,
        position_m=position_m,
        derived_velocity_mps=derived.velocity_mps,
        derived_acceleration_mps2=derived.acceleration_mps2,
        source_velocity_mps=source_velocity_mps,
        source_acceleration_mps2=source_acceleration_mps2,
        numeric_channels=(
            NumericChannel(
                name="source_speed_mps",
                units="m/s",
                role=ChannelRole.SOURCE,
                values=source_speed_mps,
            ),
            NumericChannel(
                name="derived_speed_mps",
                units="m/s",
                role=ChannelRole.DERIVED,
                values=derived.speed_mps,
            ),
            NumericChannel(
                name="heading_rad",
                units="rad",
                role=ChannelRole.DERIVED,
                values=derived.heading_rad,
            ),
            NumericChannel(
                name="yaw_rate_radps",
                units="rad/s",
                role=ChannelRole.DERIVED,
                values=derived.yaw_rate_radps,
            ),
            NumericChannel(
                name="curvature_per_m",
                units="1/m",
                role=ChannelRole.DERIVED,
                values=derived.curvature_per_m,
            ),
            NumericChannel(
                name="lane_or_region_id",
                units=None,
                role=ChannelRole.CONTEXT,
                values=_column_array(rows, "lane_id"),
            ),
            NumericChannel(
                name="length_smoothed_m",
                units="m",
                role=ChannelRole.AUDIT_ONLY,
                values=_column_array(rows, "length_m"),
            ),
            NumericChannel(
                name="width_smoothed_m",
                units="m",
                role=ChannelRole.AUDIT_ONLY,
                values=_column_array(rows, "width_m"),
            ),
        ),
        metadata={
            "adapter_id": manifest.adapter_id,
            "adapter_version": manifest.adapter_version,
            "coordinate_origin": "top_left_of_reference_image",
            "source_schema": "tgsim_foggy_bottom_table_7",
        },
    )
    quality = assess_track_quality(track, policy=config.quality_policy)
    return track.model_copy(update={"quality": quality})
####


__all__ = ["build_tgsim_track"]
