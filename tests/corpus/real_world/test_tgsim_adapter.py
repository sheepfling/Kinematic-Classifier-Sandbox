from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinematic_classifier_sandbox.corpus.real_world.adapters.tgsim import (
    load_tgsim_foggy_bottom_csv,
)
from kinematic_classifier_sandbox.corpus.real_world.adapters.tgsim_contracts import (
    DuplicateTimestampPolicy,
    InvalidRowPolicy,
    TgsimFoggyBottomAdapterConfig,
    UnknownLabelPolicy,
)
from kinematic_classifier_sandbox.corpus.real_world.contracts import ChannelRole


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tgsim_foggy_bottom_minimal.csv"
_HEADER = (
    "id,time,xloc_kf,yloc_kf,lane_kf,speed_kf_x,speed_kf_y,"
    "acceleration_kf_x,acceleration_kf_y,length_smoothed,width_smoothed,"
    "type_most_common"
)


def _write_csv(path: Path, rows: tuple[str, ...], *, header: str = _HEADER) -> Path:
    path.write_text("\n".join((header, *rows, "")), encoding="utf-8")
    return path
####


def test_adapter_loads_verified_foggy_bottom_schema() -> None:
    result = load_tgsim_foggy_bottom_csv(_FIXTURE_PATH)

    assert result.summary.rows_read == 12
    assert result.summary.tracks_loaded == 3
    assert {track.labels.normalized_class for track in result.tracks} == {
        "passenger_car",
        "bus",
        "truck",
    }
    assert result.manifest.license_id == "us-public-domain"
    assert result.manifest.doi == "10.21949/1404230"

    car_track = result.tracks[0]
    channels = {channel.name: channel for channel in car_track.numeric_channels}
    assert car_track.provenance.track_id == "10"
    assert car_track.position_m.shape == (4, 3)
    assert car_track.source_velocity_mps is not None
    assert np.allclose(car_track.source_velocity_mps[:, 0], 1.0)
    assert channels["lane_or_region_id"].role is ChannelRole.CONTEXT
    assert channels["length_smoothed_m"].role is ChannelRole.AUDIT_ONLY
    assert channels["width_smoothed_m"].role is ChannelRole.AUDIT_ONLY
    assert car_track.quality is not None
    assert car_track.quality.sample_count == 4
####


def test_adapter_groups_reused_track_ids_by_run(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "runs.csv",
        (
            "5,0.0,0.0,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3,1",
            "5,0.1,0.1,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3,1",
            "5,0.0,10.0,0.0,2,0.5,0.0,0.0,0.0,8.0,2.3,7,2",
            "5,0.1,10.05,0.0,2,0.5,0.0,0.0,0.0,8.0,2.3,7,2",
        ),
        header=f"{_HEADER},run_index",
    )

    result = load_tgsim_foggy_bottom_csv(path)

    assert result.summary.track_groups_seen == 2
    assert result.summary.tracks_loaded == 2
    assert [track.provenance.run_id for track in result.tracks] == ["1", "2"]
    assert len({track.provenance.split_group_id for track in result.tracks}) == 2
####


def test_unknown_label_requires_explicit_policy(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "unknown.csv",
        (
            "1,0.0,0.0,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,99",
            "1,0.1,0.1,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,99",
        ),
    )

    with pytest.raises(ValueError, match="unknown TGSIM native label"):
        load_tgsim_foggy_bottom_csv(path)

    skipped = load_tgsim_foggy_bottom_csv(
        path,
        config=TgsimFoggyBottomAdapterConfig(
            unknown_label_policy=UnknownLabelPolicy.SKIP
        ),
    )
    assert skipped.summary.tracks_loaded == 0
    assert skipped.summary.tracks_skipped_unknown_label == 1

    preserved = load_tgsim_foggy_bottom_csv(
        path,
        config=TgsimFoggyBottomAdapterConfig(
            unknown_label_policy=UnknownLabelPolicy.PRESERVE
        ),
    )
    assert preserved.tracks[0].labels.normalized_class == "unknown_tgsim_type_99"
    assert preserved.tracks[0].labels.evidence.value == "weak"
####


def test_duplicate_timestamp_policy_is_explicit(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "duplicates.csv",
        (
            "1,0.0,0.0,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3",
            "1,0.0,9.0,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3",
            "1,0.1,9.1,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3",
        ),
    )

    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_tgsim_foggy_bottom_csv(path)

    kept_last = load_tgsim_foggy_bottom_csv(
        path,
        config=TgsimFoggyBottomAdapterConfig(
            duplicate_timestamp_policy=DuplicateTimestampPolicy.KEEP_LAST
        ),
    )
    assert kept_last.summary.duplicate_timestamps_resolved == 1
    assert kept_last.tracks[0].position_m[0, 0] == 9.0
####


def test_adapter_accepts_hyphenated_report_headers(tmp_path: Path) -> None:
    hyphenated_header = _HEADER.replace("_", "-")
    path = _write_csv(
        tmp_path / "hyphenated.csv",
        (
            "1,0.0,0.0,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3",
            "1,0.1,0.1,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3",
        ),
        header=hyphenated_header,
    )

    result = load_tgsim_foggy_bottom_csv(path)

    assert result.summary.tracks_loaded == 1
    assert result.tracks[0].labels.normalized_class == "passenger_car"
####


def test_invalid_rows_are_never_silently_dropped(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "invalid.csv",
        (
            "1,0.0,0.0,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3",
            "1,0.1,not-a-number,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3",
            "1,0.2,0.2,0.0,1,1.0,0.0,0.0,0.0,4.5,1.8,3",
        ),
    )

    with pytest.raises(ValueError, match="invalid TGSIM row 3"):
        load_tgsim_foggy_bottom_csv(path)

    result = load_tgsim_foggy_bottom_csv(
        path,
        config=TgsimFoggyBottomAdapterConfig(
            invalid_row_policy=InvalidRowPolicy.SKIP
        ),
    )
    assert result.summary.rows_skipped_invalid == 1
    assert result.summary.tracks_loaded == 1
####
