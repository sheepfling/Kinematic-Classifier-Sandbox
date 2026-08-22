from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from pydantic import ValidationError

from kinematic_classifier_sandbox.corpus.real_world.contracts import (
    ChannelRole,
    DatasetManifest,
    LabelEvidence,
    NormalizedTrack,
    NumericChannel,
    SourceAsset,
    TrackLabels,
    TrackProvenance,
)


def _provenance() -> TrackProvenance:
    return TrackProvenance(
        dataset_id="dataset",
        source_asset_id="trajectory_csv",
        recording_id="recording",
        run_id="run",
        track_id="track",
        location_id="location",
        split_group_id="dataset:recording:run:track",
    )
####


def _labels() -> TrackLabels:
    return TrackLabels(
        native_label="3",
        normalized_class="passenger_car",
        mobility_family="conventional_steering",
        operating_domain="urban_road",
        evidence=LabelEvidence.NATIVE,
    )
####


def _track(**overrides: object) -> NormalizedTrack:
    timestamps_s = np.asarray((0.0, 0.1, 0.2), dtype=np.float64)
    position_m = np.asarray(
        ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
        dtype=np.float64,
    )
    values: dict[str, object] = {
        "provenance": _provenance(),
        "labels": _labels(),
        "coordinate_frame": "local_xy_m",
        "timestamps_s": timestamps_s,
        "position_m": position_m,
        "derived_velocity_mps": np.asarray(
            ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        "derived_acceleration_mps2": np.zeros((3, 3), dtype=np.float64),
        "metadata": {"source": "fixture"},
    }
    values.update(overrides)
    return NormalizedTrack.model_validate(values)
####


def test_normalized_track_rejects_nonmonotonic_times() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        _track(timestamps_s=np.asarray((0.0, 0.1, 0.1), dtype=np.float64))
####


def test_normalized_track_rejects_channel_length_mismatch() -> None:
    channel = NumericChannel(
        name="speed_mps",
        units="m/s",
        role=ChannelRole.DERIVED,
        values=np.asarray((1.0, 1.0), dtype=np.float64),
    )
    with pytest.raises(ValidationError, match="wrong sample count"):
        _track(numeric_channels=(channel,))
####


def test_normalized_track_freezes_arrays_and_metadata() -> None:
    source_times = np.asarray((0.0, 0.1, 0.2), dtype=np.float64)
    track = _track(timestamps_s=source_times)
    source_times[0] = 99.0

    assert track.timestamps_s[0] == 0.0
    assert not track.timestamps_s.flags.writeable
    with pytest.raises(ValueError):
        track.timestamps_s[0] = 99.0
    with pytest.raises(TypeError):
        track.metadata["source"] = "changed"
####


def test_normalized_track_schema_and_json_dump_include_numeric_arrays() -> None:
    track = _track()

    schema = NormalizedTrack.model_json_schema()
    payload = track.model_dump(mode="json")

    assert schema["properties"]["position_m"]["type"] == "array"
    assert payload["timestamps_s"] == [0.0, 0.1, 0.2]
    assert payload["position_m"][1] == [0.1, 0.0, 0.0]
    assert payload["metadata"] == {"source": "fixture"}
####


def test_dataset_manifest_rejects_duplicate_asset_ids() -> None:
    duplicate_assets = (
        SourceAsset(
            asset_id="trajectory_csv",
            title="first",
            download_url="https://example.test/first.csv",
            media_type="text/csv",
        ),
        SourceAsset(
            asset_id="trajectory_csv",
            title="second",
            download_url="https://example.test/second.csv",
            media_type="text/csv",
        ),
    )
    with pytest.raises(ValidationError, match="must be unique"):
        DatasetManifest(
            dataset_id="dataset",
            title="title",
            version="1",
            publisher="publisher",
            citation="citation",
            license_id="license",
            license_url="https://example.test/license",
            landing_page_url="https://example.test/dataset",
            accessed_on=date(2026, 8, 22),
            adapter_id="adapter",
            adapter_version="1",
            coordinate_frame="local_xy_m",
            nominal_sample_interval_s=0.1,
            source_assets=duplicate_assets,
        )
####
