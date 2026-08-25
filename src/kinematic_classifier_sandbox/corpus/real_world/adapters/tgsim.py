from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from ..contracts import DatasetManifest, NormalizedTrack
from ..manifests import (
    TGSIM_FOGGY_BOTTOM_ADAPTER_ID,
    TGSIM_FOGGY_BOTTOM_ADAPTER_VERSION,
    TGSIM_FOGGY_BOTTOM_DATASET_ID,
    build_tgsim_foggy_bottom_manifest,
)
from .tgsim_contracts import (
    InvalidRowPolicy,
    TgsimFoggyBottomAdapterConfig,
    TgsimLabelCount,
    TgsimLoadResult,
    TgsimParseSummary,
    _TgsimRow,
)
from .tgsim_parsing import (
    _normalize_headers,
    _normalized_row,
    _parse_row,
    _resolve_duplicates,
    _select_track_labels,
)
from .tgsim_track import build_tgsim_track


class TgsimFoggyBottomAdapter:
    adapter_id = TGSIM_FOGGY_BOTTOM_ADAPTER_ID
    adapter_version = TGSIM_FOGGY_BOTTOM_ADAPTER_VERSION

    def __init__(
        self,
        *,
        config: TgsimFoggyBottomAdapterConfig | None = None,
        manifest: DatasetManifest | None = None,
    ) -> None:
        self._config = config or TgsimFoggyBottomAdapterConfig()
        self._manifest = manifest or build_tgsim_foggy_bottom_manifest(
            accessed_on=self._config.accessed_on
        )
        if self._manifest.dataset_id != TGSIM_FOGGY_BOTTOM_DATASET_ID:
            raise ValueError("manifest dataset_id does not describe TGSIM Foggy Bottom")
        if self._manifest.adapter_id != self.adapter_id:
            raise ValueError("manifest adapter_id does not match this adapter")
        asset_ids = {asset.asset_id for asset in self._manifest.source_assets}
        if self._config.source_asset_id not in asset_ids:
            raise ValueError("configured source_asset_id is absent from the manifest")
    ####

    @property
    def manifest(self) -> DatasetManifest:
        return self._manifest
    ####

    def load_file(self, path: str | Path) -> TgsimLoadResult:
        source_path = Path(path)
        grouped_rows: dict[tuple[str, str], list[_TgsimRow]] = {}
        rows_read = 0
        rows_skipped_invalid = 0

        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            normalized_headers = _normalize_headers(reader.fieldnames)
            for source_row_number, raw_row in enumerate(reader, start=2):
                rows_read += 1
                try:
                    row = _parse_row(
                        _normalized_row(raw_row, normalized_headers),
                        source_row_number=source_row_number,
                        fallback_run_id=self._config.recording_id,
                    )
                except ValueError as exc:
                    if self._config.invalid_row_policy is InvalidRowPolicy.ERROR:
                        raise ValueError(
                            f"invalid TGSIM row {source_row_number}: {exc}"
                        ) from exc
                    rows_skipped_invalid += 1
                    continue
                grouped_rows.setdefault((row.run_id, row.track_id), []).append(row)

        tracks: list[NormalizedTrack] = []
        tracks_skipped_short = 0
        tracks_skipped_unknown_label = 0
        duplicate_timestamps_resolved = 0
        inconsistent_label_tracks = 0
        label_counts: Counter[tuple[str, str]] = Counter()

        groups_in_source_order = sorted(
            grouped_rows.values(),
            key=lambda items: min(item.source_row_number for item in items),
        )
        for raw_track_rows in groups_in_source_order:
            try:
                track_rows, duplicate_count = _resolve_duplicates(
                    raw_track_rows,
                    policy=self._config.duplicate_timestamp_policy,
                )
                duplicate_timestamps_resolved += duplicate_count
                if len(track_rows) < self._config.minimum_samples:
                    tracks_skipped_short += 1
                    continue
                labels, inconsistent = _select_track_labels(
                    track_rows,
                    config=self._config,
                )
                inconsistent_label_tracks += int(inconsistent)
                if labels is None:
                    tracks_skipped_unknown_label += 1
                    continue
                track = build_tgsim_track(
                    track_rows,
                    labels=labels,
                    manifest=self._manifest,
                    config=self._config,
                )
            except ValueError as exc:
                run_id = raw_track_rows[0].run_id
                track_id = raw_track_rows[0].track_id
                raise ValueError(
                    f"invalid TGSIM track run={run_id!r} id={track_id!r}: {exc}"
                ) from exc
            tracks.append(track)
            label_counts[(labels.native_label, labels.normalized_class)] += 1

        label_count_rows = tuple(
            TgsimLabelCount(
                native_label=native_label,
                normalized_class=normalized_class,
                track_count=track_count,
            )
            for (native_label, normalized_class), track_count in sorted(label_counts.items())
        )
        summary = TgsimParseSummary(
            rows_read=rows_read,
            rows_skipped_invalid=rows_skipped_invalid,
            track_groups_seen=len(grouped_rows),
            tracks_loaded=len(tracks),
            tracks_skipped_short=tracks_skipped_short,
            tracks_skipped_unknown_label=tracks_skipped_unknown_label,
            duplicate_timestamps_resolved=duplicate_timestamps_resolved,
            inconsistent_label_tracks=inconsistent_label_tracks,
            label_counts=label_count_rows,
        )
        return TgsimLoadResult(
            manifest=self._manifest,
            tracks=tuple(tracks),
            summary=summary,
        )
    ####

    def load_tracks(self, path: str | Path) -> tuple[NormalizedTrack, ...]:
        return self.load_file(path).tracks
    ####
####


def load_tgsim_foggy_bottom_csv(
    path: str | Path,
    *,
    config: TgsimFoggyBottomAdapterConfig | None = None,
    manifest: DatasetManifest | None = None,
) -> TgsimLoadResult:
    adapter = TgsimFoggyBottomAdapter(config=config, manifest=manifest)
    return adapter.load_file(path)
####


__all__ = [
    "TgsimFoggyBottomAdapter",
    "load_tgsim_foggy_bottom_csv",
]
