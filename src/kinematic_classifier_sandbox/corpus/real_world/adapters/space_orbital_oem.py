"""NASA ISS OEM Product 4 adapter for the bounded SPACE-ORB validation fixture."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

import numpy as np

from ..contracts import (
    LabelEvidence,
    NormalizedTrack,
    TrackLabels,
    TrackProvenance,
    TrackQuality,
)
from ..corpus_contracts import (
    CorpusDatasetMetadata,
    CorpusTrajectory,
    CorpusTrajectoryMetadata,
    ObservationModality,
    PhysicalDomain,
    TimeBasis,
)
from ..kinematics import differentiate_vectors
from .space_orbital_oem_metadata import (
    NASA_ISS_ADAPTER_ID,
    NASA_ISS_ADAPTER_VERSION,
    NASA_ISS_CREATION_DATE,
    NASA_ISS_DATASET_ID,
    NASA_ISS_FIRST_STATE_EPOCH,
    NASA_ISS_FIXTURE_URI,
    NASA_ISS_LAST_STATE_EPOCH,
    NASA_ISS_ORIGINAL_SOURCE_ASSET_ID,
    NASA_ISS_ORIGINAL_SOURCE_URL,
    NASA_ISS_SOURCE_ASSET_ID,
    NASA_ISS_SOURCE_SHA256,
    build_nasa_iss_corpus_metadata,
)
from .space_orbital_oem_parsing import (
    FloatArray,
    OemExtract,
    parse_oem_file,
    records_to_si_arrays,
)


_COSPAR_WITH_SEPARATE_PIECE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<year>\d{4})-(?P<number>\d{3})-(?P<piece>[A-Z0-9]+)$"
)


def canonicalize_cospar_id(value: str) -> str:
    """Normalize ``YYYY-NNN-PIECE`` to the canonical ``YYYY-NNNPIECE`` form."""

    normalized = value.strip().upper()
    match = _COSPAR_WITH_SEPARATE_PIECE.fullmatch(normalized)
    if match is not None:
        return (
            f"{match.group('year')}-{match.group('number')}"
            f"{match.group('piece')}"
        )
    return normalized
####


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
####


class NasaIssOemCorpusAdapter:
    """Load the exact public-domain NASA ISS OEM Wave 1 validation fixture."""

    adapter_id = NASA_ISS_ADAPTER_ID
    adapter_version = NASA_ISS_ADAPTER_VERSION

    def __init__(self) -> None:
        self._corpus_metadata = build_nasa_iss_corpus_metadata()
    ####

    @property
    def corpus_metadata(self) -> CorpusDatasetMetadata:
        return self._corpus_metadata
    ####

    def load_corpus(self, path: str | Path) -> tuple[CorpusTrajectory, ...]:
        source_path = Path(path)
        actual_sha256 = _sha256(source_path)
        if actual_sha256 != NASA_ISS_SOURCE_SHA256:
            raise ValueError(
                "NASA ISS OEM fixture SHA-256 mismatch: "
                f"expected {NASA_ISS_SOURCE_SHA256}, got {actual_sha256}"
            )

        extract = parse_oem_file(source_path)
        self._validate_exact_fixture(extract)
        timestamps_s, position_m, source_velocity_mps = records_to_si_arrays(
            extract.records
        )
        derived_velocity_mps = differentiate_vectors(timestamps_s, position_m)
        derived_acceleration_mps2 = differentiate_vectors(
            timestamps_s,
            derived_velocity_mps,
        )
        return (
            self._build_corpus_trajectory(
                extract=extract,
                timestamps_s=timestamps_s,
                position_m=position_m,
                source_velocity_mps=source_velocity_mps,
                derived_velocity_mps=derived_velocity_mps,
                derived_acceleration_mps2=derived_acceleration_mps2,
            ),
        )
    ####

    @staticmethod
    def _validate_exact_fixture(extract: OemExtract) -> None:
        expected_values = {
            "version": (extract.header.version, "2.0"),
            "originator": (extract.header.originator, "NASA/JSC/FOD/TOPO"),
            "object_name": (extract.metadata.object_name, "ISS"),
            "source_object_id": (extract.metadata.source_object_id, "1998-067-A"),
            "center_name": (extract.metadata.center_name, "Earth"),
            "reference_frame": (extract.metadata.reference_frame, "EME2000"),
            "time_system": (extract.metadata.time_system, "UTC"),
        }
        for field_name, (actual, expected) in expected_values.items():
            if actual != expected:
                raise ValueError(
                    f"unexpected NASA ISS OEM {field_name}: "
                    f"expected {expected!r}, got {actual!r}"
                )
        if extract.header.creation_date_utc != NASA_ISS_CREATION_DATE:
            raise ValueEror(
                "unexpected NASA ISS OEM creation date: "
                f"expected {NASA_ISS_CREATION_DATE.isoformat()}, "
                f"got {extract.header.creation_date_utc.isoformat()}"
            )
        if len(extract.records) != 13:
            raise ValueError(
                "the bounded NASA ISS fixture must contain exactly 13 state records"
            )
        if extract.records[0].epoch_utc != NASA_ISS_FIRST_STATE_EPOCH:
            raise ValueError("unexpected first epoch in the bounded NASA ISS fixture")
        if extract.records[-1].epoch_utc != NASA_ISS_LAST_STATE_EPOCH:
            raise ValueError("unexpected last epoch in the bounded NASA ISS fixture")
        timestamps_s, _, _ = records_to_si_arrays(extract.records)
        if not np.allclose(np.diff(timestamps_s), 240.0, rtol=0.0, atol=1.0e-9):
            raise ValueError("the bounded NASA ISS fixture must use a 240-second cadence")
    ####

    def _build_corpus_trajectory(
        self,
        *,
        extract: OemExtract,
        timestamps_s: FloatArray,
        position_m: FloatArray,
        source_velocity_mps: FloatArray,
        derived_velocity_mps: FloatArray,
        derived_acceleration_mps2: FloatArray,
    ) -> CorpusTrajectory:
        physical_object_id = canonicalize_cospar_id(extract.metadata.source_object_id)
        split_group_id = f"physical_object:{physical_object_id}"
        start_epoch = extract.records[0].epoch_utc
        stop_epoch = extract.records[-1].epoch_utc
        trajectory_id = (
            f"nasa-topo-iss:{start_epoch.isoformat()}:{stop_epoch.isoformat()}"
        )
        frame = self._corpus_metadata.canonical_frame
        metadata = CorpusTrajectoryMetadata(
            trajectory_id=trajectory_id,
            domain=PhysicalDomain.SPACE,
            time_basis=TimeBasis.UNIX_UTC_SECONDS,
            frame=frame,
            observation_modalities=(ObservationModality.OTHER,),
            subject_id=physical_object_id,
            platform_type="crewed_space_station",
            platform_subtype="International Space Station",
            source_metadata={
                "source_object_name": extract.metadata.object_name,
                "source_object_id": extract.metadata.source_object_id,
                "canonical_cospar_id": physical_object_id,
                "oem_creation_date": extract.header.creation_date_utc.isoformat(),
                "oem_originator": extract.header.originator,
                "source_fixture_uri": NASA_ISS_FIXTURE_URI,
                "original_source_url": NASA_ISS_ORIGINAL_SOURCE_URL,
            },
            domain_extensions={
                "central_body": "Earth",
                "orbital_regime": "LEO",
                "state_source": "NASA TOPO operational predicted OEM",
                "state_role": "estimate",
                "value_basis": "operational_prediction",
                "source_time_system": "UTC",
                "source_frame": "EME2000",
                "source_velocity_available": True,
                "source_acceleration_available": False,
                "propagation_required": False,
                "validation_fixture_only": True,
            },
        )
        track = NormalizedTrack(
            provenance=TrackProvenance(
                dataset_id=NASA_ISS_DATASET_ID,
                source_asset_id=NASA_ISS_SOURCE_ASSET_ID,
                recording_id=(
                    "ISS.OEM_J2K_EPH:"
                    f"{extract.header.creation_date_utc.isoformat()}"
                ),
                run_id="NASA-TOPO-forecast-2022-04-27",
                track_id=trajectory_id,
                location_id="earth_orbit",
                split_group_id=split_group_id,
                source_row_start=42,
                source_row_end=54,
            ),
            labels=TrackLabels(
                native_label=extract.metadata.object_name,
                normalized_class="persistent_orbit",
                mobility_family="orbital_spacecraft",
                operating_domain="space",
                evidence=LabelEvidence.DERIVED,
                is_proxy=False,
                notes=(
                    "Persistent-orbit is lane-normalized scope, not a source behavior label.",
                    "ISS identity is grouping/audit evidence, not a classifier feature.",
                ),
            ),
            coordinate_frame="EME2000",
            timestamps_s=timestamps_s,
            position_m=position_m,
            derived_velocity_mps=derived_velocity_mps,
            derived_acceleration_mps2=derived_acceleration_mps2,
            source_velocity_mps=source_velocity_mps,
            source_acceleration_mps2=None,
            quality=self._quality(
                timestamps_s=timestamps_s,
                position_m=position_m,
                source_velocity_mps=source_velocity_mps,
                derived_velocity_mps=derived_velocity_mps,
            ),
            metadata={
                "frame_kind": "eci",
                "source_time_system": "UTC",
                "state_role": "estimate",
                "value_basis": "operational_prediction",
                "physical_object_id": physical_object_id,
                "identity_access_class": "grouping_only",
                "classifier_eligible": False,
                "validation_fixture_only": True,
                "redistribution_eligible": True,
                "is_proxy": False,
            },
        )
        return CorpusTrajectory(
            dataset=self._corpus_metadata,
            metadata=metadata,
            trajectory=track,
        )
    ####

    @staticmethod
    def _quality(
        *,
        timestamps_s: FloatArray,
        position_m: FloatArray,
        source_velocity_mps: FloatArray,
        derived_velocity_mps: FloatArray,
    ) -> TrackQuality:
        dt_s = np.diff(timestamps_s)
        median_dt_s = float(np.median(dt_s))
        velocity_error_mps = derived_velocity_mps - source_velocity_mps
        return TrackQuality(
            sample_count=int(timestamps_s.shape[0]),
            duration_s=float(timestamps_s[-1] - timestamps_s[0]),
            median_dt_s=median_dt_s,
            max_dt_s=float(np.max(dt_s)),
            gap_count=int(np.sum(dt_s > (1.5 * median_dt_s))),
            max_position_step_m=float(
                np.max(np.linalg.norm(np.diff(position_m, axis=0), axis=1))
            ),
            source_velocity_rmse_mps=float(
                np.sqrt(np.mean(np.square(velocity_error_mps)))
            ),
            source_acceleration_rmse_mps2=None,
            findings=(
                "Source position and velocity are preserved in EME2000.",
                "Derived velocity and acceleration use the shared Product 4 derivative.",
                "The 48-minute bounded arc has a regular 240-second cadence.",
                "The operational prediction remains an estimate, not measurement truth.",
                "This bounded fixture is validation-only and not classifier-study eligible.",
            ),
        )
    ####
####


__all__ = [
    "NASA_ISS_ADAPTER_ID",
    "NASA_ISS_ADAPTER_VERSION",
    "NASA_ISS_DATASET_ID",
    "NASA_ISS_FIXTURE_URI",
    "NASA_ISS_ORIGINAL_SOURCE_ASSET_ID",
    "NASA_ISS_ORIGINAL_SOURCE_URL",
    "NASA_ISS_SOURCE_ASSET_ID",
    "NASA_ISS_SOURCE_SHA256",
    "NasaIssOemCorpusAdapter",
    "canonicalize_cospar_id",
]
