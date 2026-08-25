from __future__ import annotations

import numpy as np

from kinematic_classifier_sandbox.corpus.real_world.contracts import (
    LabelEvidence,
    NormalizedTrack,
    TrackLabels,
    TrackProvenance,
)
from kinematic_classifier_sandbox.corpus.real_world.kinematics import derive_planar_kinematics
from kinematic_classifier_sandbox.corpus.real_world.quality import (
    TrackQualityPolicy,
    assess_track_quality,
)


def _build_track(timestamps_s: np.ndarray, position_m: np.ndarray) -> NormalizedTrack:
    derived = derive_planar_kinematics(timestamps_s, position_m)
    return NormalizedTrack(
        provenance=TrackProvenance(
            dataset_id="dataset",
            source_asset_id="asset",
            recording_id="recording",
            run_id="run",
            track_id="track",
            location_id="location",
            split_group_id="dataset:recording:run:track",
        ),
        labels=TrackLabels(
            native_label="3",
            normalized_class="passenger_car",
            mobility_family="conventional_steering",
            operating_domain="urban_road",
            evidence=LabelEvidence.NATIVE,
        ),
        coordinate_frame="local_xy_m",
        timestamps_s=timestamps_s,
        position_m=position_m,
        derived_velocity_mps=derived.velocity_mps,
        derived_acceleration_mps2=derived.acceleration_mps2,
        source_velocity_mps=derived.velocity_mps,
        source_acceleration_mps2=derived.acceleration_mps2,
    )
####


def test_quality_reports_sampling_gap() -> None:
    timestamps_s = np.asarray((0.0, 0.1, 0.5), dtype=np.float64)
    position_m = np.asarray(
        ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.5, 0.0, 0.0)),
        dtype=np.float64,
    )
    track = _build_track(timestamps_s, position_m)

    quality = assess_track_quality(
        track,
        policy=TrackQualityPolicy(nominal_sample_interval_s=0.1, gap_multiplier=2.0),
    )

    assert quality.gap_count == 1
    assert quality.source_velocity_rmse_mps == 0.0
    assert any("sample gap" in finding for finding in quality.findings)
####


def test_planar_kinematics_exposes_heading_and_curvature() -> None:
    timestamps_s = np.asarray((0.0, 0.1, 0.2, 0.3), dtype=np.float64)
    position_m = np.asarray(
        ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.05, 0.0), (0.25, 0.15, 0.0)),
        dtype=np.float64,
    )

    derived = derive_planar_kinematics(timestamps_s, position_m)

    assert derived.velocity_mps.shape == (4, 3)
    assert derived.acceleration_mps2.shape == (4, 3)
    assert derived.heading_rad.shape == (4,)
    assert derived.yaw_rate_radps.shape == (4,)
    assert derived.curvature_per_m.shape == (4,)
    assert np.any(np.abs(derived.curvature_per_m) > 0.0)
####
