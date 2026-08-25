from __future__ import annotations

from pathlib import Path

from ..contracts import NormalizedTrack
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
from .tgsim import TgsimFoggyBottomAdapter
from .tgsim_contracts import TgsimFoggyBottomAdapterConfig


class TgsimFoggyBottomCorpusAdapter:
    """Expose TGSIM through the domain-neutral real-world corpus interface."""

    adapter_id = TgsimFoggyBottomAdapter.adapter_id
    adapter_version = TgsimFoggyBottomAdapter.adapter_version

    def __init__(
        self,
        *,
        config: TgsimFoggyBottomAdapterConfig | None = None,
    ) -> None:
        self._track_adapter = TgsimFoggyBottomAdapter(config=config)
        frame = CoordinateFrameMetadata(
            frame_id=self._track_adapter.manifest.coordinate_frame,
            kind=CoordinateFrameKind.LOCAL_CARTESIAN,
            axes_description=(
                "Source-defined planar x/y meters with z=0 in the normalized trajectory"
            ),
            origin_description="Top-left of the TGSIM Foggy Bottom reference image",
            notes=(
                "Absolute source coordinates are provenance/context, not baseline "
                "classifier features.",
            ),
        )
        self._corpus_metadata = CorpusDatasetMetadata(
            dataset_manifest=self._track_adapter.manifest,
            domains=(PhysicalDomain.LAND,),
            observation_modalities=(ObservationModality.OPTICAL_TRACKING,),
            native_coordinate_frame="TGSIM image-referenced planar metric coordinates",
            canonical_frame=frame,
            time_basis=TimeBasis.RELATIVE_SECONDS,
            source_type="overhead_video_trajectory_extraction",
            extensions={
                "road_context": "urban_intersection",
                "baseline_feature_policy": "kinematics_only",
            },
        )
    ####

    @property
    def corpus_metadata(self) -> CorpusDatasetMetadata:
        return self._corpus_metadata
    ####

    def load_corpus(self, path: str | Path) -> tuple[CorpusTrajectory, ...]:
        tracks = self._track_adapter.load_tracks(path)
        return tuple(self._wrap_track(track) for track in tracks)
    ####

    def _wrap_track(self, track: NormalizedTrack) -> CorpusTrajectory:
        provenance = track.provenance
        labels = track.labels
        metadata = CorpusTrajectoryMetadata(
            trajectory_id=provenance.split_group_id,
            domain=PhysicalDomain.LAND,
            time_basis=TimeBasis.RELATIVE_SECONDS,
            frame=self._corpus_metadata.canonical_frame,
            observation_modalities=(ObservationModality.OPTICAL_TRACKING,),
            subject_id=provenance.track_id,
            platform_type=labels.normalized_class,
            source_metadata={
                "recording_id": provenance.recording_id,
                "run_id": provenance.run_id,
                "source_asset_id": provenance.source_asset_id,
            },
            domain_extensions={
                "mobility_family": labels.mobility_family,
                "operating_domain": labels.operating_domain,
            },
        )
        return CorpusTrajectory(
            dataset=self._corpus_metadata,
            metadata=metadata,
            trajectory=track,
        )
    ####
####


__all__ = ["TgsimFoggyBottomCorpusAdapter"]
