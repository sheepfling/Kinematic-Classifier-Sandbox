"""Build immutable Product 4 snapshot manifests from external episode assets."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .episode_contracts import AssetReference, TrajectoryEpisodeManifest
from .portfolio import (
    _EVIDENCE_RANK,
    CorpusSnapshotManifest,
    SnapshotEpisodeReference,
    SourceEvidenceState,
    SourceRegistry,
    _verify_episode_assets,
    write_snapshot_manifest,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_token(*, adapter_id: str, adapter_version: str) -> str:
    return f"{adapter_id}:{adapter_version}"


def build_snapshot_manifest(
    registry: SourceRegistry,
    episode_paths: Iterable[str | Path],
    *,
    snapshot_root: str | Path,
    snapshot_id: str,
    created_at: datetime,
    require_prepared_sources: bool = False,
    notes: Iterable[str] = (),
) -> CorpusSnapshotManifest:
    """Build a snapshot manifest while retaining all source/episode boundaries.

    Episode manifests and their assets are intentionally supplied from an external snapshot
    root. This function does not copy source bytes, generate classifier assets, or advance a
    registry evidence state. ``require_prepared_sources`` is an explicit promotion guard for
    callers that want to build a classifier-eligible snapshot rather than a validation snapshot.
    """

    root = Path(snapshot_root).resolve()
    if not snapshot_id:
        raise ValueError("snapshot_id must not be empty")
    loaded: list[tuple[TrajectoryEpisodeManifest, Path, str]] = []
    registry_sources = {source.source_dataset_id: source for source in registry.sources}
    for raw_path in episode_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"episode manifest does not exist: {path}")
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"episode manifest must be inside snapshot_root: {path}"
            ) from error
        episode = TrajectoryEpisodeManifest.model_validate_json(path.read_text(encoding="utf-8"))
        _verify_episode_assets(root, episode)
        if episode.corpus_snapshot_id != snapshot_id:
            raise ValueError(
                f"episode {episode.episode_id!r} belongs to snapshot "
                f"{episode.corpus_snapshot_id!r}, not {snapshot_id!r}"
            )
        source = registry_sources.get(episode.source_dataset_id)
        if source is None:
            raise ValueError(f"episode references unknown source dataset: {episode.source_dataset_id}")
        if source.lane != episode.corpus_sublane:
            raise ValueError(f"episode lane does not match source: {episode.episode_id}")
        source_artifact_ids = {artifact.artifact_id for artifact in source.artifacts}
        if not set(episode.source_artifact_ids).issubset(source_artifact_ids):
            raise ValueError(f"episode artifacts are not declared by source: {episode.episode_id}")
        if (
            require_prepared_sources
            and _EVIDENCE_RANK[source.evidence_state]
            < _EVIDENCE_RANK[SourceEvidenceState.PREPARED]
        ):
            raise ValueError(
                f"source is not prepared for classifier snapshot: {source.source_dataset_id}"
            )
        loaded.append((episode, path, relative_path))

    if not loaded:
        raise ValueError("at least one episode manifest is required")
    loaded.sort(key=lambda item: item[0].episode_id)
    episode_ids = [episode.episode_id for episode, _, _ in loaded]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode manifest IDs must be unique")

    selected_sources = {
        episode.source_dataset_id: registry_sources[episode.source_dataset_id]
        for episode, _, _ in loaded
    }
    source_artifact_ids = tuple(
        sorted({artifact_id for episode, _, _ in loaded for artifact_id in episode.source_artifact_ids})
    )
    adapter_versions = tuple(
        sorted(
            {
                _adapter_token(
                    adapter_id=source.adapter_id,
                    adapter_version=source.adapter_version,
                )
                for source in selected_sources.values()
            }
        )
    )
    references = tuple(
        SnapshotEpisodeReference(
            episode_id=episode.episode_id,
            lane=episode.corpus_sublane,
            source_dataset_id=episode.source_dataset_id,
            manifest=AssetReference(
                path=relative_path,
                media_type="application/json",
                sha256=_sha256_file(path),
            ),
        )
        for episode, path, relative_path in loaded
    )
    return CorpusSnapshotManifest(
        snapshot_id=snapshot_id,
        registry_id=registry.registry_id,
        created_at=created_at,
        episodes=references,
        source_artifact_ids=source_artifact_ids,
        adapter_versions=adapter_versions,
        notes=tuple(notes),
    )


__all__ = ["build_snapshot_manifest", "write_snapshot_manifest"]
