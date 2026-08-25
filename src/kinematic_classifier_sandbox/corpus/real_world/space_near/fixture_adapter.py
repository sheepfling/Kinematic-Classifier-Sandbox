from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from ..contracts import (
    DatasetManifest,
    LabelEvidence,
    NormalizedTrack,
    SourceAsset,
    TrackLabels,
    TrackProvenance,
)
from ..corpus_contracts import (
    CoordinateFrameKind,
    CoordinateFrameMetadata,
    CorpusDatasetMetadata,
    CorpusTrajectory,
    CorpusTrajectoryMetadata,
    ObservationModality,
    PhysicalDomain,
    TimeBasis,
)
from ..kinematics import differentiate_vectors
from ..quality import TrackQualityPolicy, assess_track_quality
from .fixture_models import (
    EmbeddedEpisode,
    EmbeddedFixture,
    EmbeddedStateView,
    SpaceNearFixturePortfolio,
    SpaceNearFixtureValidation,
    SpaceNearSourceSpec,
)


@dataclass(frozen=True, slots=True)
class SpaceNearMissionFixtureAdapter:
    fixture: EmbeddedFixture

    adapter_id: ClassVar[str] = "space_near_repository_fixture"
    adapter_version: ClassVar[str] = "0.1.0"

    @property
    def corpus_metadata(self) -> CorpusDatasetMetadata:
        source = self.fixture.source
        frame = self.fixture.episode.analysis_frame
        coordinate_frame = CoordinateFrameMetadata(
            frame_id=frame.frame_id,
            kind=CoordinateFrameKind.ECEF,
            axes_description="Earth-centered, Earth-fixed X/Y/Z position in metres",
            origin_description="Earth center",
            authority=frame.crs_or_datum,
            epoch=None,
            vertical_datum=None,
            notes=frame.unresolved_ambiguities,
        )
        return CorpusDatasetMetadata(
            dataset_manifest=DatasetManifest(
                dataset_id=source.source_dataset_id,
                title=source.title,
                version=source.version,
                publisher=source.publisher,
                citation=source.citation,
                doi=None,
                license_id=source.license_id,
                license_url=source.license_url,
                landing_page_url=source.landing_page_url,
                accessed_on=source.accessed_on,
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                coordinate_frame=coordinate_frame.frame_id,
                nominal_sample_interval_s=source.nominal_sample_interval_s,
                source_assets=(
                    SourceAsset(
                        asset_id=source.source_asset_id,
                        title=source.source_asset_title,
                        download_url=source.source_asset_url,
                        media_type=source.source_asset_media_type,
                        sha256=source.source_asset_sha256,
                    ),
                ),
                notes=(
                    "Raw source bytes are not redistributed by this fixture tranche.",
                    "The repository fixture is a bounded derived excerpt.",
                    source.claim_boundary,
                ),
            ),
            domains=(PhysicalDomain.SPACE,),
            observation_modalities=(ObservationModality.OTHER,),
            native_coordinate_frame="source-native state retained in embedded fixture",
            canonical_frame=coordinate_frame,
            time_basis=TimeBasis.RELATIVE_SECONDS,
            source_type=source.source_type,
            extensions={
                "corpus_sublane": "space_near",
                "portfolio_role": source.portfolio_role,
                "launch_region_id": source.launch_region_id,
            },
        )
    ####

    def load_corpus(self, path: str | Path) -> tuple[CorpusTrajectory, ...]:
        if not Path(path).exists():
            raise ValueError("adapter path must identify a fixture portfolio source")
        validation = validate_embedded_fixture(self.fixture)
        timestamps_s, position_m = _analysis_arrays(self.fixture.episode.analysis_view)
        velocity_mps = differentiate_vectors(timestamps_s, position_m)
        acceleration_mps2 = differentiate_vectors(timestamps_s, velocity_mps)
        episode = self.fixture.episode
        source = self.fixture.source
        track = NormalizedTrack(
            provenance=TrackProvenance(
                dataset_id=source.source_dataset_id,
                source_asset_id=source.source_asset_id,
                recording_id=_grouping_value(episode, "source_recording"),
                run_id=episode.episode_id,
                track_id=episode.episode_id,
                location_id=source.launch_region_id,
                split_group_id=_grouping_value(episode, "physical_platform"),
            ),
            labels=TrackLabels(
                native_label=episode.default_motion_regime,
                normalized_class=episode.default_motion_regime,
                mobility_family="rocket",
                operating_domain="space_near",
                evidence=LabelEvidence.DERIVED,
                notes=(
                    "Evidence-bearing assertions remain in the embedded episode record.",
                    "The aggregate class summarizes the bounded episode only.",
                ),
            ),
            coordinate_frame=episode.analysis_frame.frame_id,
            speed_axis_count=3,
            timestamps_s=timestamps_s,
            position_m=position_m,
            derived_velocity_mps=velocity_mps,
            derived_acceleration_mps2=acceleration_mps2,
            source_velocity_mps=None,
            source_acceleration_mps2=None,
            numeric_channels=(),
            categorical_channels=(),
            quality=None,
            metadata={
                "episode_id": episode.episode_id,
                "mission_id": episode.mission_id,
                "object_id": episode.object_id,
                "state_role": episode.analysis_view.state_role,
                "analysis_state_view_id": episode.analysis_view.state_view_id,
                "quality_disposition": validation.quality_disposition,
                "absolute_start_time": episode.absolute_start_time,
                "absolute_end_time": episode.absolute_end_time,
                "portfolio_role": source.portfolio_role,
            },
        )
        quality = assess_track_quality(
            track,
            policy=TrackQualityPolicy(
                nominal_sample_interval_s=source.nominal_sample_interval_s,
                gap_multiplier=2.5,
                max_position_step_m=100_000.0,
            ),
        )
        track = track.model_copy(update={"quality": quality})
        metadata = self.corpus_metadata
        return (
            CorpusTrajectory(
                dataset=metadata,
                metadata=CorpusTrajectoryMetadata(
                    trajectory_id=episode.episode_id,
                    domain=PhysicalDomain.SPACE,
                    time_basis=TimeBasis.RELATIVE_SECONDS,
                    frame=metadata.canonical_frame,
                    observation_modalities=(ObservationModality.OTHER,),
                    subject_id=episode.object_id,
                    platform_type="sounding_rocket",
                    platform_subtype=source.platform_subtype,
                    source_metadata={
                        "source_dataset_id": source.source_dataset_id,
                        "provider": source.publisher,
                        "source_type": source.source_type,
                        "source_asset_sha256": source.source_asset_sha256,
                    },
                    domain_extensions={
                        "corpus_sublane": "space_near",
                        "mission_id": episode.mission_id,
                        "episode_id": episode.episode_id,
                        "state_role": episode.analysis_view.state_role,
                        "portfolio_role": source.portfolio_role,
                    },
                ),
                trajectory=track,
            ),
        )
    ####
####


def _view_digest(view: EmbeddedStateView) -> str:
    payload = json.dumps(
        {"columns": view.columns, "rows": view.rows},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
####


def _column_index(view: EmbeddedStateView, column: str) -> int:
    try:
        return view.columns.index(column)
    except ValueError as error:
        raise ValueError(f"state view is missing required column {column!r}") from error
    ####
####


def _analysis_arrays(view: EmbeddedStateView) -> tuple[np.ndarray, np.ndarray]:
    elapsed_index = _column_index(view, "elapsed_s")
    x_index = _column_index(view, "position_x_m")
    y_index = _column_index(view, "position_y_m")
    z_index = _column_index(view, "position_z_m")
    validity_indices = (
        _column_index(view, "position_valid_x"),
        _column_index(view, "position_valid_y"),
        _column_index(view, "position_valid_z"),
    )
    forbidden = {
        "mission_id",
        "object_id",
        "provider",
        "source_dataset_id",
        "source_asset_id",
    }
    if forbidden.intersection(view.columns):
        raise ValueError("analysis state view must not contain identity columns")
    for row in view.rows:
        if any(row[index] is not True for index in validity_indices):
            raise ValueError("analysis state view contains an invalid position sample")
    timestamps_s = np.asarray([float(row[elapsed_index]) for row in view.rows])
    position_m = np.asarray(
        [
            (float(row[x_index]), float(row[y_index]), float(row[z_index]))
            for row in view.rows
        ],
        dtype=np.float64,
    )
    return timestamps_s, position_m
####


def _grouping_value(episode: EmbeddedEpisode, namespace: str) -> str:
    matches = tuple(
        grouping_key.opaque_value
        for grouping_key in episode.grouping_keys
        if grouping_key.namespace == namespace
    )
    if len(matches) != 1:
        raise ValueError(f"expected one grouping key for namespace {namespace!r}")
    return matches[0]
####


def validate_embedded_fixture(fixture: EmbeddedFixture) -> SpaceNearFixtureValidation:
    episode = fixture.episode
    if episode.analysis_frame.axes != ("x", "y", "z"):
        raise ValueError("analysis frame must use x/y/z axes")
    if _view_digest(episode.source_native_view) != episode.source_native_view.content_sha256:
        raise ValueError("source-native view hash mismatch")
    if _view_digest(episode.analysis_view) != episode.analysis_view.content_sha256:
        raise ValueError("analysis view hash mismatch")
    processing_step_ids = {step.identifier for step in episode.processing_steps}
    if len(processing_step_ids) != len(episode.processing_steps):
        raise ValueError("processing step IDs must be unique")
    for view in (episode.source_native_view, episode.analysis_view):
        unknown_steps = set(view.processing_step_ids).difference(processing_step_ids)
        if unknown_steps:
            raise ValueError(f"state view references unknown processing steps: {unknown_steps}")
    channel_ids = set(episode.source_native_view.channel_ids) | set(
        episode.analysis_view.channel_ids
    )
    for assertion in episode.label_assertions:
        for dependency in assertion.dependency_channel_ids:
            root_channel = dependency.rsplit(".", 1)[0]
            if dependency not in channel_ids and root_channel not in channel_ids:
                raise ValueError(
                    f"label assertion references unknown channel {dependency!r}"
                )
    timestamps_s, position_m = _analysis_arrays(episode.analysis_view)
    if not np.all(np.diff(timestamps_s) > 0.0):
        raise ValueError("analysis elapsed time must be strictly increasing")
    if not np.all(np.isfinite(position_m)):
        raise ValueError("analysis ECEF position must be finite")
    return SpaceNearFixtureValidation(
        episode_id=episode.episode_id,
        source_dataset_id=episode.source_dataset_id,
        sample_count=len(episode.analysis_view.rows),
        state_view_count=2,
        label_assertion_count=len(episode.label_assertions),
        quality_disposition=episode.quality_disposition,
    )
####


def _portfolio_paths(path: str | Path) -> tuple[Path, ...]:
    portfolio_path = Path(path)
    if portfolio_path.is_file():
        return (portfolio_path,)
    if not portfolio_path.is_dir():
        raise ValueError("fixture portfolio path must be a file or directory")
    paths = tuple(sorted(portfolio_path.glob("*.json.gz")))
    if not paths:
        paths = tuple(sorted(portfolio_path.glob("*.json")))
    if not paths:
        raise ValueError("fixture portfolio directory contains no JSON fixtures")
    return paths
####


def _read_portfolio(path: Path) -> SpaceNearFixturePortfolio:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = stream.read()
    else:
        payload = path.read_text(encoding="utf-8")
    return SpaceNearFixturePortfolio.model_validate_json(payload)
####


def load_space_near_fixture_definitions(path: str | Path) -> tuple[EmbeddedFixture, ...]:
    fixtures: list[EmbeddedFixture] = []
    for portfolio_path in _portfolio_paths(path):
        fixtures.extend(_read_portfolio(portfolio_path).fixtures)
    fixture_ids = tuple(fixture.fixture_id for fixture in fixtures)
    episode_ids = tuple(fixture.episode.episode_id for fixture in fixtures)
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("fixture_id values must be unique across the portfolio directory")
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode_id values must be unique across the portfolio directory")
    return tuple(fixtures)
####


def load_space_near_fixture_portfolio(
    path: str | Path,
) -> tuple[CorpusTrajectory, ...]:
    trajectories: list[CorpusTrajectory] = []
    for fixture in load_space_near_fixture_definitions(path):
        adapter = SpaceNearMissionFixtureAdapter(fixture=fixture)
        trajectories.extend(adapter.load_corpus(path))
    return tuple(trajectories)
####


__all__ = [
    "EmbeddedEpisode",
    "EmbeddedFixture",
    "EmbeddedStateView",
    "SpaceNearFixturePortfolio",
    "SpaceNearFixtureValidation",
    "SpaceNearMissionFixtureAdapter",
    "SpaceNearSourceSpec",
    "load_space_near_fixture_definitions",
    "load_space_near_fixture_portfolio",
    "validate_embedded_fixture",
]
